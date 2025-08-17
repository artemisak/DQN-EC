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
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch

# Import the required functions from supplementary module
from supplementary import prepare_speaker, prepare_listener, create_hypergraph

# Define a named tuple for the replay buffer - now storing hypergraphs
Transition = namedtuple('Transition',
                        ('hypergraph', 'speaker_action', 'listener_action',
                         'next_hypergraph', 'speaker_reward', 'listener_reward', 'done'))


# Custom replay buffer for hypergraph-based multi-agent DQN
class HypergraphReplayBuffer:
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
    exp_name: str = "ma_dqn_hypergraph_gat"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ma_dqn_hypergraph_gat"
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

    # GAT specific arguments
    gat_hidden_dim: int = 64
    """hidden dimension for GAT layers"""
    gat_heads: int = 4
    """number of attention heads in GAT"""
    gat_dropout: float = 0.1
    """dropout rate for GAT layers"""


# Base GAT module for processing hypergraphs
class HypergraphGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Third GAT layer
        x = self.gat3(x, edge_index)

        # Global pooling to get graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        return x


# QNetwork for the speaker agent with GAT
class SpeakerGATQNetwork(nn.Module):
    def __init__(self, action_dim, gat_hidden_dim=64, gat_heads=4, dropout=0.1):
        super().__init__()

        # GAT for processing hypergraph
        self.gat = HypergraphGAT(
            input_dim=1,  # Based on hypergraph structure: x=[n, 1]
            hidden_dim=gat_hidden_dim,
            output_dim=128,
            heads=gat_heads,
            dropout=dropout
        )

        # Action value head
        self.action_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for better stability"""
        for layer in self.action_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, hypergraph_batch):
        # Process hypergraph with GAT
        if isinstance(hypergraph_batch, Data):
            # Single graph
            graph_features = self.gat(
                hypergraph_batch.x,
                hypergraph_batch.edge_index,
                hypergraph_batch.edge_attr if hasattr(hypergraph_batch, 'edge_attr') else None
            )
        else:
            # Batch of graphs
            graph_features = self.gat(
                hypergraph_batch.x,
                hypergraph_batch.edge_index,
                hypergraph_batch.edge_attr if hasattr(hypergraph_batch, 'edge_attr') else None,
                hypergraph_batch.batch
            )

        # Get Q-values for actions
        return self.action_head(graph_features)


# QNetwork for the listener agent with GAT
class ListenerGATQNetwork(nn.Module):
    def __init__(self, action_dim, gat_hidden_dim=64, gat_heads=4, dropout=0.1):
        super().__init__()

        # GAT for processing hypergraph
        self.gat = HypergraphGAT(
            input_dim=1,  # Based on hypergraph structure: x=[n, 1]
            hidden_dim=gat_hidden_dim,
            output_dim=128,
            heads=gat_heads,
            dropout=dropout
        )

        # Action value head with additional processing for movement decisions
        self.action_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for better stability"""
        for layer in self.action_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, hypergraph_batch):
        # Process hypergraph with GAT
        if isinstance(hypergraph_batch, Data):
            # Single graph
            graph_features = self.gat(
                hypergraph_batch.x,
                hypergraph_batch.edge_index,
                hypergraph_batch.edge_attr if hasattr(hypergraph_batch, 'edge_attr') else None
            )
        else:
            # Batch of graphs
            graph_features = self.gat(
                hypergraph_batch.x,
                hypergraph_batch.edge_index,
                hypergraph_batch.edge_attr if hasattr(hypergraph_batch, 'edge_attr') else None,
                hypergraph_batch.batch
            )

        # Get Q-values for actions
        return self.action_head(graph_features)


def process_observations_to_hypergraph(observations, speaker_agent, listener_agent):
    """Convert raw observations to hypergraph representation"""
    # Process speaker observation
    speaker_obs = observations[speaker_agent]
    speaker_data = prepare_speaker(speaker_obs)

    # Process listener observation (first 8 values only)
    listener_obs = observations[listener_agent][:8]
    listener_data = prepare_listener(listener_obs)

    # Create shared hypergraph
    hypergraph = create_hypergraph([speaker_data, listener_data])

    return hypergraph


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def update_target_network(target_net, online_net, tau):
    """Soft update of target network parameters: θ' ← τθ + (1 − τ)θ'"""
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

    # Get action dimensions
    speaker_action_dim = env.action_space(speaker_agent).n
    listener_action_dim = env.action_space(listener_agent).n

    print(f"Speaker action dim: {speaker_action_dim}")
    print(f"Listener action dim: {listener_action_dim}")
    print(f"Using GAT with {args.gat_heads} heads and hidden dim {args.gat_hidden_dim}")

    # Create GAT-based Q-networks for each agent
    speaker_q_net = SpeakerGATQNetwork(
        speaker_action_dim,
        args.gat_hidden_dim,
        args.gat_heads,
        args.gat_dropout
    ).to(device)
    speaker_target_net = SpeakerGATQNetwork(
        speaker_action_dim,
        args.gat_hidden_dim,
        args.gat_heads,
        args.gat_dropout
    ).to(device)
    speaker_target_net.load_state_dict(speaker_q_net.state_dict())
    speaker_optimizer = optim.Adam(speaker_q_net.parameters(), lr=args.learning_rate)

    listener_q_net = ListenerGATQNetwork(
        listener_action_dim,
        args.gat_hidden_dim,
        args.gat_heads,
        args.gat_dropout
    ).to(device)
    listener_target_net = ListenerGATQNetwork(
        listener_action_dim,
        args.gat_hidden_dim,
        args.gat_heads,
        args.gat_dropout
    ).to(device)
    listener_target_net.load_state_dict(listener_q_net.state_dict())
    listener_optimizer = optim.Adam(listener_q_net.parameters(), lr=args.learning_rate)

    # Create shared hypergraph replay buffer
    replay_buffer = HypergraphReplayBuffer(args.buffer_size)

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

    # Convert initial observations to hypergraph
    current_hypergraph = process_observations_to_hypergraph(observations, speaker_agent, listener_agent)

    for global_step in range(args.total_timesteps):
        # Epsilon-greedy exploration
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)

        # Move hypergraph to device
        current_hypergraph_device = Data(
            x=current_hypergraph.x.to(device),
            edge_index=current_hypergraph.edge_index.to(device),
            edge_attr=current_hypergraph.edge_attr.to(device) if hasattr(current_hypergraph, 'edge_attr') else None,
            pos=current_hypergraph.pos.to(device) if hasattr(current_hypergraph, 'pos') else None
        )

        # Select actions
        actions = {}

        # Speaker action selection
        if random.random() < epsilon:
            actions[speaker_agent] = env.action_space(speaker_agent).sample()
        else:
            with torch.no_grad():
                q_values = speaker_q_net(current_hypergraph_device)
                actions[speaker_agent] = torch.argmax(q_values, dim=1).item()

        # Update speaker action histogram for visualization
        speaker_actions_hist[actions[speaker_agent]] += 1

        # Listener action selection
        if random.random() < epsilon:
            actions[listener_agent] = env.action_space(listener_agent).sample()
        else:
            with torch.no_grad():
                q_values = listener_q_net(current_hypergraph_device)
                actions[listener_agent] = torch.argmax(q_values, dim=1).item()

        # Update listener action histogram for visualization
        listener_actions_hist[actions[listener_agent]] += 1

        # Execute actions
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Convert next observations to hypergraph
        next_hypergraph = process_observations_to_hypergraph(next_observations, speaker_agent, listener_agent)

        # Store transition in shared replay buffer
        done = terminations[speaker_agent] or truncations[speaker_agent] or \
               terminations[listener_agent] or truncations[listener_agent]

        replay_buffer.push(
            current_hypergraph,
            actions[speaker_agent],
            actions[listener_agent],
            next_hypergraph,
            rewards[speaker_agent],
            rewards[listener_agent],
            done
        )

        # Update metrics
        episode_reward += (rewards[speaker_agent] + rewards[listener_agent]) / 2
        episode_length += 1

        # Check if episode ended
        if any(terminations.values()) or any(truncations.values()):
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
            current_hypergraph = process_observations_to_hypergraph(observations, speaker_agent, listener_agent)
        else:
            # Update hypergraph for next step
            current_hypergraph = next_hypergraph
            observations = next_observations

        # Training logic
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            if len(replay_buffer) >= args.batch_size:
                # Sample batch from replay buffer
                transitions = replay_buffer.sample(args.batch_size)
                batch_transition = Transition(*zip(*transitions))

                # Create batched graphs for current and next states
                current_graphs = [g for g in batch_transition.hypergraph]
                next_graphs = [g for g in batch_transition.next_hypergraph]

                # Batch the graphs
                current_batch = Batch.from_data_list(current_graphs).to(device)
                next_batch = Batch.from_data_list(next_graphs).to(device)

                # Prepare action and reward tensors
                speaker_actions = torch.LongTensor(batch_transition.speaker_action).unsqueeze(1).to(device)
                listener_actions = torch.LongTensor(batch_transition.listener_action).unsqueeze(1).to(device)
                speaker_rewards = torch.FloatTensor(batch_transition.speaker_reward).to(device)
                listener_rewards = torch.FloatTensor(batch_transition.listener_reward).to(device)
                dones = torch.FloatTensor(batch_transition.done).to(device)

                # Train speaker agent
                with torch.no_grad():
                    next_speaker_q_values = speaker_target_net(next_batch)
                    next_speaker_q_max = next_speaker_q_values.max(1)[0]
                    speaker_targets = speaker_rewards + args.gamma * next_speaker_q_max * (1 - dones)

                current_speaker_q_values = speaker_q_net(current_batch)
                speaker_q_values = current_speaker_q_values.gather(1, speaker_actions).squeeze(1)
                speaker_loss = F.mse_loss(speaker_q_values, speaker_targets)

                speaker_optimizer.zero_grad()
                speaker_loss.backward()
                torch.nn.utils.clip_grad_norm_(speaker_q_net.parameters(), max_norm=1.0)
                speaker_optimizer.step()

                # Train listener agent
                with torch.no_grad():
                    next_listener_q_values = listener_target_net(next_batch)
                    next_listener_q_max = next_listener_q_values.max(1)[0]
                    listener_targets = listener_rewards + args.gamma * next_listener_q_max * (1 - dones)

                current_listener_q_values = listener_q_net(current_batch)
                listener_q_values = current_listener_q_values.gather(1, listener_actions).squeeze(1)
                listener_loss = F.mse_loss(listener_q_values, listener_targets)

                listener_optimizer.zero_grad()
                listener_loss.backward()
                torch.nn.utils.clip_grad_norm_(listener_q_net.parameters(), max_norm=1.0)
                listener_optimizer.step()

                # Log training metrics
                if global_step % 100 == 0:
                    if args.track:
                        wandb.log({
                            "speaker_loss": speaker_loss.item(),
                            "speaker_q_values": speaker_q_values.mean().item(),
                            "listener_loss": listener_loss.item(),
                            "listener_q_values": listener_q_values.mean().item(),
                            "global_step": global_step
                        }, step=global_step)

            # Log general training metrics
            if global_step % 100 == 0:
                if len(episode_rewards) > 0:
                    window_size = min(10, len(episode_rewards))
                    avg_reward = np.mean(episode_rewards[-window_size:])

                    if args.track:
                        wandb.log({
                            "average_reward_last_10": avg_reward,
                            "epsilon": epsilon,
                            "steps_per_second": int(global_step / (time.time() - start_time)),
                            "global_step": global_step
                        }, step=global_step)

        # Update target networks with soft updates
        if global_step % args.target_network_frequency == 0:
            update_target_network(speaker_target_net, speaker_q_net, args.tau)
            update_target_network(listener_target_net, listener_q_net, args.tau)

    # Save models
    if args.save_model:
        os.makedirs(f"models/{run_name}", exist_ok=True)
        torch.save(speaker_q_net.state_dict(), f"models/{run_name}/speaker_model.pt")
        torch.save(listener_q_net.state_dict(), f"models/{run_name}/listener_model.pt")
        print(f"Models saved to models/{run_name}")

        # Save metadata for evaluation
        metadata = {
            "speaker_action_dim": speaker_action_dim,
            "listener_action_dim": listener_action_dim,
            "gat_hidden_dim": args.gat_hidden_dim,
            "gat_heads": args.gat_heads,
            "gat_dropout": args.gat_dropout,
            "time": time.time()
        }
        np.save(f"models/{run_name}/metadata.npy", metadata)

    env.close()

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()