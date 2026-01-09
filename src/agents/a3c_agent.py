"""
A3C (Asynchronous Advantage Actor-Critic) Multi-Agent Implementation
Simplified single-threaded version for demonstration
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List
from collections import deque
import os

from .base_agent import BaseAgent


class A3CNetwork(nn.Module):
    """Actor-Critic network for A3C"""

    def __init__(self, obs_dim: int, action_dims: List[int], hidden_dim: int = 256):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor heads
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for action_dim in action_dims
        ])

        # Critic
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)

        # Actor logits
        action_logits = [head(features) for head in self.actor_heads]

        # Critic value
        value = self.critic(features)

        return action_logits, value

    def get_action(self, x, deterministic=False):
        action_logits, value = self.forward(x)
        actions = []
        log_probs = []

        for logits in action_logits:
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            actions.append(action)
            log_probs.append(log_prob)

        return actions, torch.stack(log_probs).sum(0), value


class A3CAgent(BaseAgent):
    """Multi-Agent A3C implementation (single-threaded)"""

    def __init__(self, config: Dict[str, Any], obs_dim: int, action_dims: List[int]):
        super().__init__(config)

        self.obs_dim = obs_dim
        self.action_dims = action_dims

        # Hyperparameters (Improved)
        training_config = config.get('training', {})
        self.lr = training_config.get('learning_rate', 3e-4)  # Increased from 1e-4
        self.gamma = training_config.get('gamma', 0.99)
        self.value_coef = 0.5
        self.entropy_coef = training_config.get('ent_coef', 0.02)  # Increased
        self.max_grad_norm = training_config.get('max_grad_norm', 0.5)
        self.n_steps = 50  # Increased from 20 for more stable gradients

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = A3CNetwork(obs_dim, action_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_loss': []
        }

    def train(self, env, total_timesteps: int) -> Dict[str, Any]:
        """Train A3C agent"""
        num_agents = env.num_lgvs
        observations, _ = env.reset()

        episode_reward = np.zeros(num_agents)
        episode_length = 0

        # Storage for n-step returns
        states_buffer = []
        actions_buffer = []
        rewards_buffer = []
        log_probs_buffer = []
        values_buffer = []
        dones_buffer = []

        for step in range(total_timesteps):
            # Select actions
            obs_tensor = torch.FloatTensor(
                np.array([observations[i] for i in range(num_agents)])
            ).to(self.device)

            with torch.no_grad():
                actions_list, log_probs, values = self.model.get_action(obs_tensor)

            # Convert actions
            actions = {i: torch.stack([a[i] for a in actions_list]).cpu().numpy()
                      for i in range(num_agents)}

            # Step environment
            next_observations, rewards, dones, truncated, info = env.step(actions)

            # Store experience
            states_buffer.append(observations)
            actions_buffer.append(actions)
            rewards_buffer.append([rewards[i] for i in range(num_agents)])
            log_probs_buffer.append(log_probs.cpu().numpy())
            values_buffer.append(values.squeeze(-1).cpu().numpy())
            dones_buffer.append([dones[i] for i in range(num_agents)])

            episode_reward += np.array([rewards[i] for i in range(num_agents)])
            episode_length += 1

            observations = next_observations

            # Update after n steps or episode end
            if len(states_buffer) >= self.n_steps or dones.get('__all__', False):
                # Bootstrap value
                if not dones.get('__all__', False):
                    obs_tensor = torch.FloatTensor(
                        np.array([observations[i] for i in range(num_agents)])
                    ).to(self.device)
                    with torch.no_grad():
                        _, _, bootstrap_value = self.model.get_action(obs_tensor)
                    bootstrap_value = bootstrap_value.squeeze(-1).cpu().numpy()
                else:
                    bootstrap_value = np.zeros(num_agents)

                # Compute returns
                returns = self._compute_returns(
                    rewards_buffer,
                    values_buffer,
                    bootstrap_value,
                    dones_buffer
                )

                # Update policy
                self._update_policy(
                    states_buffer,
                    actions_buffer,
                    log_probs_buffer,
                    returns,
                    values_buffer
                )

                # Clear buffers
                states_buffer = []
                actions_buffer = []
                rewards_buffer = []
                log_probs_buffer = []
                values_buffer = []
                dones_buffer = []

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

    def _compute_returns(self, rewards, values, bootstrap_value, dones):
        """Compute n-step returns"""
        num_agents = len(rewards[0])
        T = len(rewards)

        returns = np.zeros((T, num_agents))

        for agent_id in range(num_agents):
            R = bootstrap_value[agent_id]
            for t in reversed(range(T)):
                R = rewards[t][agent_id] + self.gamma * R * (1 - dones[t][agent_id])
                returns[t, agent_id] = R

        return returns

    def _update_policy(self, states, actions, old_log_probs, returns, values):
        """Update policy using A3C loss"""
        num_agents = len(states[0])
        T = len(states)

        # Flatten data
        states_flat = []
        actions_flat = [[] for _ in range(len(self.action_dims))]
        old_log_probs_flat = []
        returns_flat = []
        values_flat = []

        for t in range(T):
            for agent_id in range(num_agents):
                states_flat.append(states[t][agent_id])
                for action_idx in range(len(self.action_dims)):
                    actions_flat[action_idx].append(actions[t][agent_id][action_idx])
                old_log_probs_flat.append(old_log_probs[t][agent_id])
                returns_flat.append(returns[t, agent_id])
                values_flat.append(values[t][agent_id])

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states_flat)).to(self.device)
        actions_tensor = [torch.LongTensor(np.array(a)).to(self.device) for a in actions_flat]
        returns_tensor = torch.FloatTensor(np.array(returns_flat)).to(self.device)
        values_tensor = torch.FloatTensor(np.array(values_flat)).to(self.device)

        # Get current predictions
        action_logits, values_pred = self.model(states_tensor)

        # Compute log probs and entropy
        log_probs = []
        entropy = []
        for i, logits in enumerate(action_logits):
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(actions_tensor[i])
            log_probs.append(log_prob)
            entropy.append(dist.entropy())

        log_probs = torch.stack(log_probs).sum(0)
        entropy = torch.stack(entropy).sum(0)

        # Compute advantages
        advantages = returns_tensor - values_tensor.detach()

        # Actor loss
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss
        values_pred = values_pred.squeeze(-1)
        critic_loss = nn.functional.mse_loss(values_pred, returns_tensor)

        # Total loss
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log metrics
        self.training_metrics['actor_loss'].append(actor_loss.item())
        self.training_metrics['critic_loss'].append(critic_loss.item())
        self.training_metrics['entropy'].append(entropy.mean().item())
        self.training_metrics['total_loss'].append(loss.item())

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        """Predict action"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions_list, _, _ = self.model.get_action(obs_tensor, deterministic=deterministic)

        actions = torch.stack([a[0] for a in actions_list]).cpu().numpy()
        return actions, None

    def save(self, path: str):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics"""
        return {
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'actor_loss': np.mean(self.training_metrics['actor_loss'][-100:]) if self.training_metrics['actor_loss'] else 0,
            'critic_loss': np.mean(self.training_metrics['critic_loss'][-100:]) if self.training_metrics['critic_loss'] else 0,
            'entropy': np.mean(self.training_metrics['entropy'][-100:]) if self.training_metrics['entropy'] else 0
        }
