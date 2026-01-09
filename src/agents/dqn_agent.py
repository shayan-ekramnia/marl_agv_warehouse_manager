"""
DQN (Deep Q-Network) Multi-Agent Implementation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List
from collections import deque
import random
import os

from .base_agent import BaseAgent


class QNetwork(nn.Module):
    """Q-Network for DQN"""

    def __init__(self, obs_dim: int, action_dims: List[int], hidden_dim: int = 256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Separate Q-value heads for each action dimension
        self.q_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for action_dim in action_dims
        ])

    def forward(self, x):
        features = self.shared(x)
        q_values = [head(features) for head in self.q_heads]
        return q_values


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            actions,
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """Multi-Agent DQN implementation"""

    def __init__(self, config: Dict[str, Any], obs_dim: int, action_dims: List[int]):
        super().__init__(config)

        self.obs_dim = obs_dim
        self.action_dims = action_dims

        # Hyperparameters (Improved)
        training_config = config.get('training', {})
        self.lr = training_config.get('learning_rate', 1e-4)
        self.gamma = training_config.get('gamma', 0.99)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05  # Increased from 0.01 (more exploration)
        self.epsilon_decay = 0.9995  # Slower decay
        self.epsilon = self.epsilon_start
        self.batch_size = training_config.get('batch_size', 128)  # Increased
        self.target_update_freq = 500  # More frequent updates
        self.learning_starts = 500  # Start learning sooner

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = QNetwork(obs_dim, action_dims).to(self.device)
        self.target_network = QNetwork(obs_dim, action_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)

        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_metrics = {
            'loss': [],
            'q_values': [],
            'epsilon': []
        }
        self.steps = 0

    def train(self, env, total_timesteps: int) -> Dict[str, Any]:
        """Train DQN agent"""
        num_agents = env.num_lgvs
        observations, _ = env.reset()

        episode_reward = np.zeros(num_agents)
        episode_length = 0

        for step in range(total_timesteps):
            self.steps += 1

            # Select actions (epsilon-greedy for each agent)
            actions = {}
            for agent_id in range(num_agents):
                if random.random() < self.epsilon:
                    # Random action
                    actions[agent_id] = np.array([
                        random.randint(0, dim - 1) for dim in self.action_dims
                    ])
                else:
                    # Greedy action
                    obs_tensor = torch.FloatTensor(observations[agent_id]).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = self.q_network(obs_tensor)
                    actions[agent_id] = np.array([
                        torch.argmax(q).item() for q in q_values
                    ])

            # Step environment
            next_observations, rewards, dones, truncated, info = env.step(actions)

            # Store experiences
            for agent_id in range(num_agents):
                self.replay_buffer.push(
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_observations[agent_id],
                    dones[agent_id]
                )

            episode_reward += np.array([rewards[i] for i in range(num_agents)])
            episode_length += 1

            observations = next_observations

            # Update Q-network
            if self.steps >= self.learning_starts and len(self.replay_buffer) >= self.batch_size:
                loss = self._update_q_network()
                self.training_metrics['loss'].append(loss)

            # Update target network
            if self.steps % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_metrics['epsilon'].append(self.epsilon)

            # Handle episode end
            if dones.get('__all__', False):
                self.episode_rewards.append(np.mean(episode_reward))
                self.episode_lengths.append(episode_length)

                observations, _ = env.reset()
                episode_reward = np.zeros(num_agents)
                episode_length = 0

        return {
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'training_metrics': self.training_metrics
        }

    def _update_q_network(self) -> float:
        """Update Q-network using sampled batch"""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values_list = self.q_network(states_tensor)

        # Get Q values for taken actions
        current_q_values = []
        for i, q_vals in enumerate(current_q_values_list):
            action_indices = torch.LongTensor([a[i] for a in actions]).to(self.device)
            current_q = q_vals.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            current_q_values.append(current_q)

        current_q_values = torch.stack(current_q_values).mean(0)

        # Target Q values
        with torch.no_grad():
            next_q_values_list = self.target_network(next_states_tensor)
            next_q_values = torch.stack([q.max(1)[0] for q in next_q_values_list]).mean(0)
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, target_q_values)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Track Q values
        self.training_metrics['q_values'].append(current_q_values.mean().item())

        return loss.item()

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        """Predict action"""
        if not deterministic and random.random() < self.epsilon:
            # Random action
            actions = np.array([random.randint(0, dim - 1) for dim in self.action_dims])
        else:
            # Greedy action
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            actions = np.array([torch.argmax(q).item() for q in q_values])

        return actions, None

    def save(self, path: str):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.steps = checkpoint.get('steps', 0)

    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics"""
        return {
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'loss': np.mean(self.training_metrics['loss'][-100:]) if self.training_metrics['loss'] else 0,
            'q_values': np.mean(self.training_metrics['q_values'][-100:]) if self.training_metrics['q_values'] else 0,
            'epsilon': self.epsilon
        }
