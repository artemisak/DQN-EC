import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pettingzoo.mpe import simple_speaker_listener_v4
import os

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DQN(nn.Module):
    """Deep Q-Network"""

    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience Replay Buffer"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DoubleDQNAgent:
    """Double DQN Agent"""

    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-networks
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(10000)

        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update_freq = 100
        self.train_step = 0

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update(self):
        """Update the Q-network using Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Use online network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(num_episodes=1000, max_steps=25):
    """Train Double DQN agents"""
    env = simple_speaker_listener_v4.parallel_env(max_cycles=max_steps, continuous_actions=False)

    # Initialize agents
    agents = {}
    for agent_name in env.possible_agents:
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)
        agents[agent_name] = DoubleDQNAgent(obs_space.shape[0], act_space.n)

    episode_rewards = []

    print("Starting training...")
    print(f"Device: {agents['speaker_0'].device}")

    for episode in range(num_episodes):
        observations, infos = env.reset(seed=SEED + episode)
        episode_reward = {agent: 0 for agent in env.possible_agents}

        for step in range(max_steps):
            actions = {}
            for agent_name in env.agents:
                actions[agent_name] = agents[agent_name].select_action(observations[agent_name])

            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # Store transitions
            for agent_name in env.agents:
                done = terminations[agent_name] or truncations[agent_name]
                agents[agent_name].replay_buffer.push(
                    observations[agent_name],
                    actions[agent_name],
                    rewards[agent_name],
                    next_observations[agent_name],
                    done
                )
                episode_reward[agent_name] += rewards[agent_name]

            # Update networks
            for agent_name in agents:
                agents[agent_name].update()

            observations = next_observations

            if not env.agents:
                break

        # Decay epsilon
        for agent_name in agents:
            agents[agent_name].decay_epsilon()

        total_reward = sum(episode_reward.values())
        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agents['speaker_0'].epsilon:.3f}")

    env.close()

    # Save models
    os.makedirs("models", exist_ok=True)
    for agent_name, agent in agents.items():
        torch.save(agent.q_network.state_dict(), f"models/{agent_name}_ddqn.pth")
    print("\nModels saved to 'models/' directory")

    return agents


if __name__ == "__main__":
    # Train agents
    trained_agents = train(num_episodes=1000, max_steps=25)