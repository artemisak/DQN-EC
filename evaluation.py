import os
import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4


# Import model definitions from our implementation
# These classes are identical to those in our main script
class SpeakerQNetwork(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class ListenerQNetwork(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)


def find_latest_model():
    """Find the latest model in the models directory"""
    if not os.path.exists("models"):
        raise FileNotFoundError("No 'models' directory found. Train a model first.")

    run_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
    if not run_dirs:
        raise FileNotFoundError("No run directories found in 'models'. Train a model first.")

    # Find the most recent run directory by looking at metadata timestamps
    latest_dir = None
    latest_time = 0

    for dir_name in run_dirs:
        metadata_path = os.path.join("models", dir_name, "metadata.npy")
        if os.path.exists(metadata_path):
            try:
                metadata = np.load(metadata_path, allow_pickle=True).item()
                if metadata["time"] > latest_time:
                    latest_time = metadata["time"]
                    latest_dir = dir_name
            except:
                # Skip if metadata can't be loaded
                continue

    # If no metadata found, use the most recently modified directory
    if latest_dir is None:
        latest_dir = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join("models", d)))

    speaker_path = os.path.join("models", latest_dir, "speaker_model.pt")
    listener_path = os.path.join("models", latest_dir, "listener_model.pt")
    metadata_path = os.path.join("models", latest_dir, "metadata.npy")

    if not os.path.exists(speaker_path) or not os.path.exists(listener_path):
        raise FileNotFoundError(f"Model files not found in {os.path.join('models', latest_dir)}")

    # Load metadata if it exists, otherwise use defaults
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
    else:
        # Default values if metadata not found
        metadata = {
            "speaker_obs_dim": None,
            "speaker_action_dim": None,
            "listener_obs_dim": None,
            "listener_action_dim": None
        }

    return speaker_path, listener_path, latest_dir, metadata


def evaluate():
    """Evaluate the latest trained model in human render mode"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find and load the latest model
    try:
        speaker_path, listener_path, run_dir, metadata = find_latest_model()
        print(f"Loading models from {run_dir}")
    except FileNotFoundError as e:
        print(e)
        return

    # Create environment with human render mode for visualization
    env = simple_speaker_listener_v4.parallel_env(render_mode="human")
    observations, _ = env.reset()

    # Get agent names
    speaker_agent = [agent for agent in env.agents if 'speaker' in agent][0]
    listener_agent = [agent for agent in env.agents if 'listener' in agent][0]

    # Get observation and action dimensions
    speaker_obs_dim = metadata["speaker_obs_dim"] if metadata["speaker_obs_dim"] is not None else np.prod(
        env.observation_space(speaker_agent).shape)
    speaker_action_dim = metadata["speaker_action_dim"] if metadata[
                                                               "speaker_action_dim"] is not None else env.action_space(
        speaker_agent).n

    listener_obs_dim = metadata["listener_obs_dim"] if metadata["listener_obs_dim"] is not None else np.prod(
        env.observation_space(listener_agent).shape)
    listener_action_dim = metadata["listener_action_dim"] if metadata[
                                                                 "listener_action_dim"] is not None else env.action_space(
        listener_agent).n

    print(f"Speaker obs dim: {speaker_obs_dim}, action dim: {speaker_action_dim}")
    print(f"Listener obs dim: {listener_obs_dim}, action dim: {listener_action_dim}")

    # Create Q-networks for each agent
    speaker_model = SpeakerQNetwork(speaker_obs_dim, speaker_action_dim).to(device)
    listener_model = ListenerQNetwork(listener_obs_dim, listener_action_dim).to(device)

    # Load model weights
    speaker_model.load_state_dict(torch.load(speaker_path, map_location=device))
    listener_model.load_state_dict(torch.load(listener_path, map_location=device))

    speaker_model.eval()
    listener_model.eval()

    # Run evaluation
    rewards = []
    num_episodes = 5

    for i in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = 0
        done = False
        episode_length = 0

        while not done:
            # Get speaker action
            obs_tensor = torch.FloatTensor(observations[speaker_agent]).unsqueeze(0).to(device)
            with torch.no_grad():
                speaker_q_values = speaker_model(obs_tensor)
            speaker_action = torch.argmax(speaker_q_values, dim=1).item()

            # Get listener action
            obs_tensor = torch.FloatTensor(observations[listener_agent]).unsqueeze(0).to(device)
            with torch.no_grad():
                listener_q_values = listener_model(obs_tensor)
            listener_action = torch.argmax(listener_q_values, dim=1).item()

            actions = {
                speaker_agent: speaker_action,
                listener_agent: listener_action
            }

            # Execute actions
            observations, rewards_dict, terminations, truncations, _ = env.step(actions)

            # Print current actions and rewards for debugging
            print(f"Speaker action: {speaker_action}, Listener action: {listener_action}")
            print(f"Rewards: {rewards_dict}")

            # Update metrics
            episode_reward += (rewards_dict[speaker_agent] + rewards_dict[listener_agent]) / 2
            episode_length += 1

            # Check if done
            done = any(terminations.values()) or any(truncations.values())

        rewards.append(episode_reward)
        print(f"Episode {i + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    print(f"Average evaluation reward over {num_episodes} episodes: {np.mean(rewards):.2f}")
    env.close()


if __name__ == "__main__":
    evaluate()
