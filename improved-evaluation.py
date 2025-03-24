import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_speaker_listener_v4


# Import model definitions from our implementation
# These classes are identical to those in our main script
class SpeakerQNetwork(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Enhanced architecture for better message encoding
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )
        
        # Message selection head - more direct path for key information
        self.action_head = torch.nn.Linear(128, action_dim)
        
        # Better initialization for more stable learning
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for better stability"""
        for layer in self.feature_extractor:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=1.0)
                torch.nn.init.constant_(layer.bias, 0.0)
        
        torch.nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        torch.nn.init.constant_(self.action_head.bias, 0.0)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.action_head(features)


class ListenerQNetwork(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, comm_dim=10):
        super().__init__()
        # Communication dimension is 10 (one-hot encoded say_0 to say_9)
        self.comm_dim = comm_dim
        self.other_dim = obs_dim - comm_dim
        
        # Process communication signal with special attention
        self.comm_net = torch.nn.Sequential(
            torch.nn.Linear(self.comm_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
        )
        
        # Process positional and velocity information
        self.pos_vel_net = torch.nn.Sequential(
            torch.nn.Linear(self.other_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
        )
        
        # Final action selection from combined information
        self.combined_net = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )
        
        # Initialize all weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for better stability"""
        for net in [self.comm_net, self.pos_vel_net]:
            for layer in net:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, gain=1.0)
                    torch.nn.init.constant_(layer.bias, 0.0)
        
        for i, layer in enumerate(self.combined_net):
            if isinstance(layer, torch.nn.Linear):
                if i == len(self.combined_net) - 1:
                    # Last layer with smaller weights
                    torch.nn.init.orthogonal_(layer.weight, gain=0.01)
                else:
                    torch.nn.init.orthogonal_(layer.weight, gain=1.0)
                torch.nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # Split observation into communication and position/velocity components
        comm = x[:, -self.comm_dim:]  # Last comm_dim elements are communication
        pos_vel = x[:, :-self.comm_dim]  # The rest are position and velocity
        
        # Process components separately
        comm_features = self.comm_net(comm)
        pos_vel_features = self.pos_vel_net(pos_vel)
        
        # Combine and decide action
        combined = torch.cat([comm_features, pos_vel_features], dim=1)
        return self.combined_net(combined)


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
            "listener_action_dim": None,
            "comm_dim": 10  # Default communication dimension
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
        
    # Get communication dimension
    comm_dim = metadata.get("comm_dim", speaker_action_dim)  # Use speaker_action_dim as default if not specified

    print(f"Speaker obs dim: {speaker_obs_dim}, action dim: {speaker_action_dim}")
    print(f"Listener obs dim: {listener_obs_dim}, action dim: {listener_action_dim}")
    print(f"Communication dimension: {comm_dim}")

    # Create Q-networks for each agent with improved architectures
    speaker_model = SpeakerQNetwork(speaker_obs_dim, speaker_action_dim).to(device)
    listener_model = ListenerQNetwork(listener_obs_dim, listener_action_dim, comm_dim).to(device)

    # Load model weights
    speaker_model.load_state_dict(torch.load(speaker_path, map_location=device))
    listener_model.load_state_dict(torch.load(listener_path, map_location=device))

    speaker_model.eval()
    listener_model.eval()

    # Run evaluation
    rewards = []
    distances = []
    num_episodes = 10
    message_counts = np.zeros(speaker_action_dim)

    for i in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = 0
        done = False
        episode_length = 0
        
        # Track final distance to target
        final_distance = None
        
        # For tracking what messages are used
        episode_messages = []

        while not done:
            # Get speaker action - no exploration during evaluation
            obs_tensor = torch.FloatTensor(observations[speaker_agent]).unsqueeze(0).to(device)
            with torch.no_grad():
                speaker_q_values = speaker_model(obs_tensor)
            speaker_action = torch.argmax(speaker_q_values, dim=1).item()
            episode_messages.append(speaker_action)
            
            # Track message usage
            message_counts[speaker_action] += 1

            # Get listener action - no exploration during evaluation
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
            print(f"Episode {i+1}, Step {episode_length+1}:")
            print(f"  Speaker message: {speaker_action}, Listener action: {listener_action}")
            print(f"  Rewards: Speaker={rewards_dict[speaker_agent]:.3f}, Listener={rewards_dict[listener_agent]:.3f}")
            
            # Try to extract distance to landmark if available
            # This is environment-specific and may need adjustment
            try:
                # Listener position and target landmark might be in the observation
                # The exact format depends on the environment
                listener_pos = observations[listener_agent][3:5]  # Typical indices for listener position
                target_pos = observations[speaker_agent][:2]     # Typical indices for target position
                distance = np.sqrt(np.sum((listener_pos - target_pos)**2))
                final_distance = distance
                print(f"  Distance to target: {distance:.3f}")
            except:
                print("  Unable to calculate distance to target")
            
            # Update metrics
            episode_reward += (rewards_dict[speaker_agent] + rewards_dict[listener_agent]) / 2
            episode_length += 1

            # Check if done
            done = any(terminations.values()) or any(truncations.values())

        rewards.append(episode_reward)
        if final_distance is not None:
            distances.append(final_distance)
            
        # Print episode summary
        print(f"\nEpisode {i + 1} Summary:")
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")
        print(f"  Messages used: {np.bincount(episode_messages)}")
        if final_distance is not None:
            print(f"  Final distance to target: {final_distance:.3f}")
        print("\n" + "-"*50 + "\n")

    # Print overall statistics
    print("\nEvaluation Results:")
    print(f"Average reward over {num_episodes} episodes: {np.mean(rewards):.2f}")
    if distances:
        print(f"Average final distance to target: {np.mean(distances):.3f}")
    
    # Print message usage statistics
    print("\nMessage Usage Statistics:")
    for i in range(speaker_action_dim):
        percentage = (message_counts[i] / message_counts.sum()) * 100 if message_counts.sum() > 0 else 0
        print(f"  Message {i}: {message_counts[i]} times ({percentage:.1f}%)")
    
    # Visualize message distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(speaker_action_dim), message_counts)
    plt.xlabel('Message ID')
    plt.ylabel('Frequency')
    plt.title('Message Usage Distribution')
    plt.xticks(range(speaker_action_dim))
    plt.savefig('message_distribution.png')
    plt.close()
    
    # Visualize reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=min(10, num_episodes))
    plt.xlabel('Episode Reward')
    plt.ylabel('Count')
    plt.title('Reward Distribution')
    plt.savefig('reward_distribution.png')
    plt.close()
    
    # If we have distances, visualize them too
    if distances:
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=min(10, len(distances)))
        plt.xlabel('Final Distance to Target')
        plt.ylabel('Count')
        plt.title('Distance Distribution')
        plt.savefig('distance_distribution.png')
        plt.close()
        
    print("Visualizations saved as PNG files.")
    env.close()


if __name__ == "__main__":
    evaluate()