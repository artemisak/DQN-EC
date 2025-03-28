import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from typing import Optional
from collections import deque, namedtuple
from pettingzoo.mpe import simple_speaker_listener_v4

# Define a named tuple for the replay buffer
Transition = namedtuple('Transition',
                        ('obs', 'action', 'next_obs', 'reward', 'done'))


# Custom replay buffer for multi-agent DQN
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


@dataclass
class Args:
    exp_name: str = "ma_dqn_speaker_listener"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ma_dqn_speaker_listener"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    total_timesteps: int = 300000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    buffer_size: int = 51200
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 50
    """the timesteps it takes to update the target network"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 5120
    """timestep to start learning"""
    train_frequency: int = 5
    """the frequency of training"""


# QNetwork for the speaker agent
class SpeakerQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)


# QNetwork for the listener agent
class ListenerQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():
    args = tyro.cli(Args)

    # Set up tracking and logging
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    env = simple_speaker_listener_v4.parallel_env(render_mode="rgb_array" if args.capture_video else None)
    observations, _ = env.reset(seed=args.seed)

    # Get agent names
    speaker_agent = [agent for agent in env.agents if 'speaker' in agent][0]
    listener_agent = [agent for agent in env.agents if 'listener' in agent][0]

    # Get observation and action dimensions
    speaker_obs_dim = np.prod(env.observation_space(speaker_agent).shape)
    speaker_action_dim = env.action_space(speaker_agent).n

    listener_obs_dim = np.prod(env.observation_space(listener_agent).shape)
    listener_action_dim = env.action_space(listener_agent).n

    print(f"Speaker obs dim: {speaker_obs_dim}, action dim: {speaker_action_dim}")
    print(f"Listener obs dim: {listener_obs_dim}, action dim: {listener_action_dim}")

    # Create Q-networks for each agent
    speaker_q_net = SpeakerQNetwork(speaker_obs_dim, speaker_action_dim).to(device)
    speaker_target_net = SpeakerQNetwork(speaker_obs_dim, speaker_action_dim).to(device)
    speaker_target_net.load_state_dict(speaker_q_net.state_dict())
    speaker_optimizer = optim.Adam(speaker_q_net.parameters(), lr=args.learning_rate)

    listener_q_net = ListenerQNetwork(listener_obs_dim, listener_action_dim).to(device)
    listener_target_net = ListenerQNetwork(listener_obs_dim, listener_action_dim).to(device)
    listener_target_net.load_state_dict(listener_q_net.state_dict())
    listener_optimizer = optim.Adam(listener_q_net.parameters(), lr=args.learning_rate)

    # Create replay buffers for each agent
    speaker_buffer = ReplayBuffer(args.buffer_size)
    listener_buffer = ReplayBuffer(args.buffer_size)

    # Metrics
    episode_rewards = []
    episode_lengths = []
    episode_reward = 0
    episode_length = 0
    episode_count = 0

    # Training visualization metrics
    speaker_actions_hist = np.zeros(speaker_action_dim)
    listener_actions_hist = np.zeros(listener_action_dim)

    # Start training
    start_time = time.time()

    for global_step in range(args.total_timesteps):
        # Epsilon-greedy exploration
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)

        # Select actions
        actions = {}

        # Speaker action selection
        if random.random() < epsilon:
            actions[speaker_agent] = env.action_space(speaker_agent).sample()
        else:
            obs_tensor = torch.FloatTensor(observations[speaker_agent]).unsqueeze(0).to(device)
            q_values = speaker_q_net(obs_tensor)
            actions[speaker_agent] = torch.argmax(q_values, dim=1).item()

        # Update speaker action histogram for visualization
        speaker_actions_hist[actions[speaker_agent]] += 1

        # Listener action selection
        if random.random() < epsilon:
            actions[listener_agent] = env.action_space(listener_agent).sample()
        else:
            obs_tensor = torch.FloatTensor(observations[listener_agent]).unsqueeze(0).to(device)
            q_values = listener_q_net(obs_tensor)
            actions[listener_agent] = torch.argmax(q_values, dim=1).item()

        # Update listener action histogram for visualization
        listener_actions_hist[actions[listener_agent]] += 1

        # Execute actions
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store transitions in replay buffers
        speaker_buffer.push(
            observations[speaker_agent],
            actions[speaker_agent],
            next_observations[speaker_agent],
            rewards[speaker_agent],
            terminations[speaker_agent] or truncations[speaker_agent]
        )

        listener_buffer.push(
            observations[listener_agent],
            actions[listener_agent],
            next_observations[listener_agent],
            rewards[listener_agent],
            terminations[listener_agent] or truncations[listener_agent]
        )

        # Update metrics
        episode_reward += (rewards[speaker_agent] + rewards[listener_agent]) / 2
        episode_length += 1

        # Check if episode ended
        done = any(terminations.values()) or any(truncations.values())
        if done:

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if args.track and episode_count % 10 == 0:
                # Log to W&B
                wandb.log({
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "speaker_reward": rewards[speaker_agent],
                    "listener_reward": rewards[listener_agent],
                    "global_step": global_step,
                    "epsilon": epsilon
                }, step=global_step)

                # Log action distributions every 10 episodes
                if np.sum(speaker_actions_hist) > 0:
                    speaker_action_probs = speaker_actions_hist / np.sum(speaker_actions_hist)
                    listener_action_probs = listener_actions_hist / np.sum(listener_actions_hist)

                    # Reset histograms
                    speaker_actions_hist = np.zeros(speaker_action_dim)
                    listener_actions_hist = np.zeros(listener_action_dim)

                    # Log as bar charts
                    wandb.log({
                        "speaker_action_distribution": wandb.plot.bar(
                            wandb.Table(data=[[i, p] for i, p in enumerate(speaker_action_probs)],
                                        columns=["action", "probability"]),
                            "action", "probability",
                            title="Speaker Action Distribution"
                        ),
                        "listener_action_distribution": wandb.plot.bar(
                            wandb.Table(data=[[i, p] for i, p in enumerate(listener_action_probs)],
                                        columns=["action", "probability"]),
                            "action", "probability",
                            title="Listener Action Distribution"
                        )
                    }, step=global_step)

            # Reset episode metrics
            episode_reward = 0
            episode_length = 0
            episode_count += 1

            # Reset environment
            observations, _ = env.reset()
        else:
            # Update observations
            observations = next_observations

        # Training logic
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            # Train speaker agent if enough samples
            if len(speaker_buffer) >= args.batch_size:
                transitions = speaker_buffer.sample(args.batch_size)
                batch = Transition(*zip(*transitions))

                # Prepare batch
                state_batch = torch.FloatTensor(np.array(batch.obs)).to(device)
                action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
                next_state_batch = torch.FloatTensor(np.array(batch.next_obs)).to(device)
                reward_batch = torch.FloatTensor(batch.reward).to(device)
                done_batch = torch.FloatTensor(batch.done).to(device)

                # Compute speaker TD target
                with torch.no_grad():
                    next_q_values = speaker_target_net(next_state_batch)
                    next_q_values_max = next_q_values.max(1)[0]
                    expected_q_values = reward_batch + args.gamma * next_q_values_max * (1 - done_batch)

                # Compute speaker loss
                current_q_values = speaker_q_net(state_batch).gather(1, action_batch).squeeze(1)
                speaker_loss = F.mse_loss(current_q_values, expected_q_values)

                # Update speaker network
                speaker_optimizer.zero_grad()
                speaker_loss.backward()
                speaker_optimizer.step()

                # Log speaker metrics
                if global_step % 100 == 0:

                    if args.track:
                        wandb.log({
                            "speaker_loss": speaker_loss.item(),
                            "speaker_q_values": current_q_values.mean().item(),
                            "global_step": global_step
                        }, step=global_step)

            # Train listener agent if enough samples
            if len(listener_buffer) >= args.batch_size:
                transitions = listener_buffer.sample(args.batch_size)
                batch = Transition(*zip(*transitions))

                # Prepare batch
                state_batch = torch.FloatTensor(np.array(batch.obs)).to(device)
                action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
                next_state_batch = torch.FloatTensor(np.array(batch.next_obs)).to(device)
                reward_batch = torch.FloatTensor(batch.reward).to(device)
                done_batch = torch.FloatTensor(batch.done).to(device)

                # Compute listener TD target
                with torch.no_grad():
                    next_q_values = listener_target_net(next_state_batch)
                    next_q_values_max = next_q_values.max(1)[0]
                    expected_q_values = reward_batch + args.gamma * next_q_values_max * (1 - done_batch)

                # Compute listener loss
                current_q_values = listener_q_net(state_batch).gather(1, action_batch).squeeze(1)
                listener_loss = F.mse_loss(current_q_values, expected_q_values)

                # Update listener network
                listener_optimizer.zero_grad()
                listener_loss.backward()
                listener_optimizer.step()

                # Log listener metrics
                if global_step % 100 == 0:

                    if args.track:
                        wandb.log({
                            "listener_loss": listener_loss.item(),
                            "listener_q_values": current_q_values.mean().item(),
                            "global_step": global_step
                        }, step=global_step)

            # Log general training metrics
            if global_step % 100 == 0:

                # Performance over time
                if len(episode_rewards) > 0:
                    window_size = min(10, len(episode_rewards))
                    avg_reward = np.mean(episode_rewards[-window_size:])

                    if args.track:
                        wandb.log({
                            "average_reward_last_10": avg_reward,
                            "epsilon": epsilon,
                            "steps_per_second": int(global_step / (time.time() - start_time)),
                            "global_step": global_step,
                            # Add heatmap of communication success
                            "communication_success": wandb.plot.scatter(
                                wandb.Table(data=[[r, l] for r, l in
                                                  zip(episode_rewards[-window_size:], episode_lengths[-window_size:])],
                                            columns=["reward", "episode_length"]),
                                "reward", "episode_length",
                                title="Communication Success vs Episode Length"
                            )
                        }, step=global_step)

        # Update target networks
        if global_step % args.target_network_frequency == 0:
            # Update speaker target network
            for target_param, param in zip(speaker_target_net.parameters(), speaker_q_net.parameters()):
                target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

            # Update listener target network
            for target_param, param in zip(listener_target_net.parameters(), listener_q_net.parameters()):
                target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

    # Save models
    if args.save_model:
        os.makedirs(f"models/{run_name}", exist_ok=True)
        torch.save(speaker_q_net.state_dict(), f"models/{run_name}/speaker_model.pt")
        torch.save(listener_q_net.state_dict(), f"models/{run_name}/listener_model.pt")
        print(f"Models saved to models/{run_name}")

        # Save metadata for evaluation
        metadata = {
            "speaker_obs_dim": speaker_obs_dim,
            "speaker_action_dim": speaker_action_dim,
            "listener_obs_dim": listener_obs_dim,
            "listener_action_dim": listener_action_dim,
            "comm_dim": 0, # Added for compatibility
            "time": time.time()
        }
        np.save(f"models/{run_name}/metadata.npy", metadata)

    env.close()

    if args.track:
        wandb.finish()

if __name__ == "__main__":
    main()
