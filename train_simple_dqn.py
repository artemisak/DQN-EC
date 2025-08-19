import os
import random
import time
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import tyro
from pettingzoo.mpe import simple_speaker_listener_v4

# Transition for replay buffer - now using raw observations
Transition = namedtuple('Transition',
                        ('speaker_obs', 'listener_obs', 'speaker_action', 'listener_action',
                         'next_speaker_obs', 'next_listener_obs',
                         'speaker_reward', 'listener_reward', 'done'))


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
    exp_name: str = "ma_dqn_fixed"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "ma_dqn_fixed"
    wandb_entity: Optional[str] = None
    save_model: bool = True

    # Algorithm arguments
    total_timesteps: int = 100000
    learning_rate: float = 5e-4
    buffer_size: int = 50000
    gamma: float = 0.99
    tau: float = 0.005  # Softer updates
    target_network_frequency: int = 100
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.3
    learning_starts: int = 5000
    train_frequency: int = 4


class SpeakerQNetwork(nn.Module):
    """Simple MLP Q-network for speaker (color selection)"""

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs):
        return self.net(obs)


class ListenerQNetwork(nn.Module):
    """MLP Q-network for listener (movement)"""

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Listener needs to process both its observation and the message
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs):
        return self.net(obs)


def normalize_observations(obs, agent_type):
    """Normalize observations to [-1, 1] range"""
    obs_tensor = torch.FloatTensor(obs)

    if agent_type == 'speaker':
        # Speaker sees goal (landmark colors), typically in [0, 1] range
        return obs_tensor * 2.0 - 1.0
    else:  # listener
        # Listener sees positions and velocities
        # Positions are typically in [-1, 1], velocities in [-2, 2]
        # Simple clipping for stability
        return torch.clamp(obs_tensor, -2.0, 2.0) / 2.0


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def update_target_network(target_net, online_net, tau):
    """Soft update of target network parameters"""
    for target_param, param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def main():
    args = tyro.cli(Args)

    # Setup
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    env = simple_speaker_listener_v4.parallel_env()
    observations, _ = env.reset(seed=args.seed)

    # Get agent names
    speaker_agent = [agent for agent in env.agents if 'speaker' in agent][0]
    listener_agent = [agent for agent in env.agents if 'listener' in agent][0]

    # Get dimensions
    speaker_obs_dim = len(observations[speaker_agent])
    listener_obs_dim = len(observations[listener_agent])
    speaker_action_dim = env.action_space(speaker_agent).n
    listener_action_dim = env.action_space(listener_agent).n

    print(f"Speaker: obs_dim={speaker_obs_dim}, action_dim={speaker_action_dim}")
    print(f"Listener: obs_dim={listener_obs_dim}, action_dim={listener_action_dim}")

    # Create Q-networks
    speaker_q_net = SpeakerQNetwork(speaker_obs_dim, speaker_action_dim).to(device)
    speaker_target_net = SpeakerQNetwork(speaker_obs_dim, speaker_action_dim).to(device)
    speaker_target_net.load_state_dict(speaker_q_net.state_dict())
    speaker_optimizer = optim.Adam(speaker_q_net.parameters(), lr=args.learning_rate)

    listener_q_net = ListenerQNetwork(listener_obs_dim, listener_action_dim).to(device)
    listener_target_net = ListenerQNetwork(listener_obs_dim, listener_action_dim).to(device)
    listener_target_net.load_state_dict(listener_q_net.state_dict())
    listener_optimizer = optim.Adam(listener_q_net.parameters(), lr=args.learning_rate)

    # Replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size)

    # Metrics
    episode_rewards = []
    episode_count = 0
    episode_reward = 0
    episode_speaker_reward = 0
    episode_listener_reward = 0
    episode_length = 0

    start_time = time.time()

    for global_step in range(args.total_timesteps):
        # Epsilon for exploration
        epsilon = linear_schedule(args.start_e, args.end_e,
                                  args.exploration_fraction * args.total_timesteps,
                                  global_step)

        # Get normalized observations
        speaker_obs = normalize_observations(observations[speaker_agent], 'speaker')
        listener_obs = normalize_observations(observations[listener_agent], 'listener')

        # Select actions
        actions = {}

        # Speaker action
        if random.random() < epsilon:
            actions[speaker_agent] = env.action_space(speaker_agent).sample()
        else:
            with torch.no_grad():
                q_values = speaker_q_net(speaker_obs.unsqueeze(0).to(device))
                actions[speaker_agent] = q_values.argmax().item()

        # Listener action
        if random.random() < epsilon:
            actions[listener_agent] = env.action_space(listener_agent).sample()
        else:
            with torch.no_grad():
                q_values = listener_q_net(listener_obs.unsqueeze(0).to(device))
                actions[listener_agent] = q_values.argmax().item()

        # Step environment
        next_observations, rewards, terminations, truncations, _ = env.step(actions)

        # Get next normalized observations
        next_speaker_obs = normalize_observations(next_observations[speaker_agent], 'speaker')
        next_listener_obs = normalize_observations(next_observations[listener_agent], 'listener')

        # Check if done
        done = terminations[speaker_agent] or truncations[speaker_agent]

        # Store transition
        replay_buffer.push(
            speaker_obs.numpy(),
            listener_obs.numpy(),
            actions[speaker_agent],
            actions[listener_agent],
            next_speaker_obs.numpy(),
            next_listener_obs.numpy(),
            rewards[speaker_agent],
            rewards[listener_agent],
            done
        )

        # Update metrics
        episode_reward += (rewards[speaker_agent] + rewards[listener_agent]) / 2
        episode_speaker_reward += rewards[speaker_agent]
        episode_listener_reward += rewards[listener_agent]
        episode_length += 1

        # Episode end
        if done:
            episode_rewards.append(episode_reward)

            if args.track:
                wandb.log({
                    "episode_reward": episode_reward,
                    "speaker_episode_reward": episode_speaker_reward,
                    "listener_episode_reward": episode_listener_reward,
                    "episode_length": episode_length,
                    "epsilon": epsilon,
                    "global_step": global_step
                })

            # Print progress
            if episode_count % 100 == 0:
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                print(f"Episode {episode_count}, Step {global_step}")
                print(f"  Avg reward (last 100): {np.mean(recent_rewards):.3f}")
                print(f"  Speaker reward: {episode_speaker_reward:.3f}")
                print(f"  Listener reward: {episode_listener_reward:.3f}")
                print(f"  Epsilon: {epsilon:.3f}")

            # Reset
            episode_reward = 0
            episode_speaker_reward = 0
            episode_listener_reward = 0
            episode_length = 0
            episode_count += 1
            observations, _ = env.reset()
        else:
            observations = next_observations

        # Training
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            if len(replay_buffer) >= args.batch_size:
                # Sample batch
                transitions = replay_buffer.sample(args.batch_size)
                batch = Transition(*zip(*transitions))

                # Convert to tensors
                speaker_obs_batch = torch.FloatTensor(batch.speaker_obs).to(device)
                listener_obs_batch = torch.FloatTensor(batch.listener_obs).to(device)
                speaker_action_batch = torch.LongTensor(batch.speaker_action).unsqueeze(1).to(device)
                listener_action_batch = torch.LongTensor(batch.listener_action).unsqueeze(1).to(device)
                next_speaker_obs_batch = torch.FloatTensor(batch.next_speaker_obs).to(device)
                next_listener_obs_batch = torch.FloatTensor(batch.next_listener_obs).to(device)
                speaker_reward_batch = torch.FloatTensor(batch.speaker_reward).to(device)
                listener_reward_batch = torch.FloatTensor(batch.listener_reward).to(device)
                done_batch = torch.FloatTensor(batch.done).to(device)

                # Train speaker
                with torch.no_grad():
                    next_q_values = speaker_q_net(next_speaker_obs_batch)
                    next_actions = next_q_values.argmax(dim=1, keepdim=True)
                    next_q_values_target = speaker_target_net(next_speaker_obs_batch)
                    next_q_value = next_q_values_target.gather(1, next_actions).squeeze()
                    target_q_value = speaker_reward_batch + args.gamma * next_q_value * (1 - done_batch)

                current_q_values = speaker_q_net(speaker_obs_batch)
                current_q_value = current_q_values.gather(1, speaker_action_batch).squeeze()

                speaker_loss = F.smooth_l1_loss(current_q_value, target_q_value)

                speaker_optimizer.zero_grad()
                speaker_loss.backward()
                torch.nn.utils.clip_grad_norm_(speaker_q_net.parameters(), 1.0)
                speaker_optimizer.step()

                # Train listener
                with torch.no_grad():
                    next_q_values = listener_q_net(next_listener_obs_batch)
                    next_actions = next_q_values.argmax(dim=1, keepdim=True)
                    next_q_values_target = listener_target_net(next_listener_obs_batch)
                    next_q_value = next_q_values_target.gather(1, next_actions).squeeze()
                    target_q_value = listener_reward_batch + args.gamma * next_q_value * (1 - done_batch)

                current_q_values = listener_q_net(listener_obs_batch)
                current_q_value = current_q_values.gather(1, listener_action_batch).squeeze()

                listener_loss = F.smooth_l1_loss(current_q_value, target_q_value)

                listener_optimizer.zero_grad()
                listener_loss.backward()
                torch.nn.utils.clip_grad_norm_(listener_q_net.parameters(), 1.0)
                listener_optimizer.step()

                # Log losses
                if global_step % 100 == 0 and args.track:
                    wandb.log({
                        "speaker_loss": speaker_loss.item(),
                        "listener_loss": listener_loss.item(),
                        "speaker_q_values": current_q_values.mean().item(),
                        "listener_q_values": listener_q_net(listener_obs_batch).mean().item(),
                        "global_step": global_step
                    })

        # Update target networks
        if global_step % args.target_network_frequency == 0:
            update_target_network(speaker_target_net, speaker_q_net, args.tau)
            update_target_network(listener_target_net, listener_q_net, args.tau)

    # Save models
    if args.save_model:
        os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
        torch.save(speaker_q_net.state_dict(), f"checkpoints/{run_name}/speaker_model.pt")
        torch.save(listener_q_net.state_dict(), f"checkpoints/{run_name}/listener_model.pt")
        print(f"Models saved to checkpoints/{run_name}")

    env.close()

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()