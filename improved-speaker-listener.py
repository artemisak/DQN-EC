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
    tau: float = 0.1
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


# Improved QNetwork for the speaker agent
# Speaker observation space: [goal_id]
# Speaker action space: [say_0, say_1, ..., say_9]
class SpeakerQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Enhanced architecture for better message encoding
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Message selection head - more direct path for key information
        self.action_head = nn.Linear(128, action_dim)
        
        # Better initialization for more stable learning
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for better stability"""
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.action_head.weight, gain=np.sqrt(2))
        nn.init.constant_(self.action_head.bias, 0.0)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.action_head(features)


# Improved QNetwork for the listener agent
# Listener observation space: [self_vel, all_landmark_rel_positions, communication]
# Listener action space: [no_action, move_left, move_right, move_down, move_up]
class ListenerQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, comm_dim):
        super().__init__()
        self.comm_dim = comm_dim
        self.other_dim = obs_dim - comm_dim
        
        # Process communication signal with special attention
        self.comm_net = nn.Sequential(
            nn.Linear(self.comm_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # Process positional and velocity information
        self.pos_vel_net = nn.Sequential(
            nn.Linear(self.other_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # Final action selection from combined information
        self.combined_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Initialize all weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for better stability"""
        for net in [self.comm_net, self.pos_vel_net, self.combined_net]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
        
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


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def update_target_network(target_net, online_net, tau):
    """Soft update of target network parameters: θ′ ← τθ + (1 − τ)θ′"""
    for target_param, param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


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
    
    # Communication dimension is based on speaker action space
    comm_dim = speaker_action_dim

    print(f"Speaker obs dim: {speaker_obs_dim}, action dim: {speaker_action_dim}")
    print(f"Listener obs dim: {listener_obs_dim}, action dim: {listener_action_dim}")
    print(f"Communication dimension: {comm_dim}")

    # Create Q-networks for each agent
    speaker_q_net = SpeakerQNetwork(speaker_obs_dim, speaker_action_dim).to(device)
    speaker_target_net = SpeakerQNetwork(speaker_obs_dim, speaker_action_dim).to(device)
    speaker_target_net.load_state_dict(speaker_q_net.state_dict())
    speaker_optimizer = optim.Adam(speaker_q_net.parameters(), lr=args.learning_rate)

    listener_q_net = ListenerQNetwork(listener_obs_dim, listener_action_dim, comm_dim).to(device)
    listener_target_net = ListenerQNetwork(listener_obs_dim, listener_action_dim, comm_dim).to(device)
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
                })

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
                    })

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
                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(speaker_q_net.parameters(), max_norm=1.0)
                speaker_optimizer.step()

                # Log speaker metrics
                if global_step % 100 == 0:

                    if args.track:
                        wandb.log({
                            "speaker_loss": speaker_loss.item(),
                            "speaker_q_values": current_q_values.mean().item(),
                            "global_step": global_step
                        })

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
                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(listener_q_net.parameters(), max_norm=1.0)
                listener_optimizer.step()

                # Log listener metrics
                if global_step % 100 == 0:

                    if args.track:
                        wandb.log({
                            "listener_loss": listener_loss.item(),
                            "listener_q_values": current_q_values.mean().item(),
                            "global_step": global_step
                        })

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
                        })

        # Update target networks with soft updates
        if global_step % args.target_network_frequency == 0:
            # Update speaker target network
            update_target_network(speaker_target_net, speaker_q_net, args.tau)
            
            # Update listener target network
            update_target_network(listener_target_net, listener_q_net, args.tau)

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
            "comm_dim": comm_dim,
            "time": time.time()
        }
        np.save(f"models/{run_name}/metadata.npy", metadata)

    env.close()

    if args.track:
        wandb.finish()

if __name__ == "__main__":
    main()
