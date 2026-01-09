"""
PPO (Proximal Policy Optimization) Multi-Agent Implementation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List
from collections import deque
import os

from .base_agent import BaseAgent


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""

    def __init__(self, obs_dim: int, action_dims: List[int], hidden_dim: int = 256):
        super().__init__()

        self.actor_shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Separate heads for each action dimension
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for action_dim in action_dims
        ])

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Actor
        actor_features = self.actor_shared(x)
        action_logits = [head(actor_features) for head in self.actor_heads]

        # Critic
        value = self.critic(x)

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


class PPOAgent(BaseAgent):
    """Multi-Agent PPO implementation"""

    def __init__(self, config: Dict[str, Any], obs_dim: int, action_dims: List[int]):
        super().__init__(config)

        self.obs_dim = obs_dim
        self.action_dims = action_dims

        # Hyperparameters (Improved for better learning)
        training_config = config.get('training', {})
        self.lr = training_config.get('learning_rate', 3e-4)
        self.gamma = training_config.get('gamma', 0.99)
        self.gae_lambda = 0.95
        self.clip_epsilon = training_config.get('clip_range', 0.2)
        self.value_coef = 0.5
        self.entropy_coef = training_config.get('ent_coef', 0.02)  # Increased default
        self.max_grad_norm = training_config.get('max_grad_norm', 0.5)
        self.n_steps = training_config.get('n_steps', 512)  # Reduced default
        self.batch_size = training_config.get('batch_size', 128)  # Increased default
        self.n_epochs = 10

        # Learning rate scheduling
        self.initial_lr = self.lr
        self.lr_decay = 0.999

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = ActorCritic(obs_dim, action_dims).to(self.device)
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
        """Train PPO agent"""
        num_agents = env.num_lgvs
        observations, _ = env.reset()

        episode_reward = np.zeros(num_agents)
        episode_length = 0

        # Storage
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []

        for step in range(total_timesteps):
            # Collect experience
            obs_tensor = torch.FloatTensor(
                np.array([observations[i] for i in range(num_agents)])
            ).to(self.device)

            with torch.no_grad():
                actions_list, log_probs, values = self.model.get_action(obs_tensor)

            # Convert actions to numpy
            actions = {i: torch.stack([a[i] for a in actions_list]).cpu().numpy()
                      for i in range(num_agents)}

            # Step environment
            next_observations, rewards, dones, truncated, info = env.step(actions)

            # Store experience
            rollout_obs.append(observations)
            rollout_actions.append(actions)
            rollout_log_probs.append(log_probs.cpu().numpy())
            rollout_values.append(values.squeeze(-1).cpu().numpy())
            rollout_rewards.append([rewards[i] for i in range(num_agents)])
            rollout_dones.append([dones[i] for i in range(num_agents)])

            episode_reward += np.array([rewards[i] for i in range(num_agents)])
            episode_length += 1

            observations = next_observations

            # Update policy
            if len(rollout_obs) >= self.n_steps or dones.get('__all__', False):
                # Compute returns and advantages
                returns, advantages = self._compute_gae(
                    rollout_rewards,
                    rollout_values,
                    rollout_dones
                )

                # Update policy
                self._update_policy(
                    rollout_obs,
                    rollout_actions,
                    rollout_log_probs,
                    returns,
                    advantages
                )

                # Clear rollout buffer
                rollout_obs = []
                rollout_actions = []
                rollout_log_probs = []
                rollout_values = []
                rollout_rewards = []
                rollout_dones = []

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

    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        num_agents = len(rewards[0])
        T = len(rewards)

        returns = np.zeros((T, num_agents))
        advantages = np.zeros((T, num_agents))

        for agent_id in range(num_agents):
            gae = 0
            next_value = 0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_non_terminal = 1.0 - dones[t][agent_id]
                else:
                    next_non_terminal = 1.0 - dones[t][agent_id]
                    next_value = values[t + 1][agent_id]

                delta = rewards[t][agent_id] + self.gamma * next_value * next_non_terminal - values[t][agent_id]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

                advantages[t, agent_id] = gae
                returns[t, agent_id] = gae + values[t][agent_id]

        return returns, advantages

    def _update_policy(self, obs, actions, old_log_probs, returns, advantages):
        """Update policy using PPO loss"""
        num_agents = len(obs[0])
        T = len(obs)

        # Flatten data
        obs_flat = []
        actions_flat = [[] for _ in range(len(self.action_dims))]
        old_log_probs_flat = []
        returns_flat = []
        advantages_flat = []

        for t in range(T):
            for agent_id in range(num_agents):
                obs_flat.append(obs[t][agent_id])
                for action_idx in range(len(self.action_dims)):
                    actions_flat[action_idx].append(actions[t][agent_id][action_idx])
                old_log_probs_flat.append(old_log_probs[t][agent_id])
                returns_flat.append(returns[t, agent_id])
                advantages_flat.append(advantages[t, agent_id])

        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs_flat)).to(self.device)
        actions_tensor = [torch.LongTensor(np.array(a)).to(self.device) for a in actions_flat]
        old_log_probs_tensor = torch.FloatTensor(np.array(old_log_probs_flat)).to(self.device)
        returns_tensor = torch.FloatTensor(np.array(returns_flat)).to(self.device)
        advantages_tensor = torch.FloatTensor(np.array(advantages_flat)).to(self.device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Update for multiple epochs
        dataset_size = len(obs_flat)
        for _ in range(self.n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get current policy predictions
                action_logits, values = self.model(obs_tensor[batch_indices])

                # Compute log probs
                log_probs = []
                entropy = []
                for i, logits in enumerate(action_logits):
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    log_prob = dist.log_prob(actions_tensor[i][batch_indices])
                    log_probs.append(log_prob)
                    entropy.append(dist.entropy())

                log_probs = torch.stack(log_probs).sum(0)
                entropy = torch.stack(entropy).sum(0)

                # PPO loss
                ratio = torch.exp(log_probs - old_log_probs_tensor[batch_indices])
                surr1 = ratio * advantages_tensor[batch_indices]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor[batch_indices]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_pred = values.squeeze(-1)
                critic_loss = nn.functional.mse_loss(value_pred, returns_tensor[batch_indices])

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
