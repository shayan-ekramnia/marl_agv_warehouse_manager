"""
Training pipeline for MARL agents
"""
import yaml
import numpy as np
from typing import Dict, Any, Optional
import os
import json
from datetime import datetime
import pickle

from ..environment.warehouse_env import WarehouseEnv
from ..agents import PPOAgent, DQNAgent, A3CAgent


class Trainer:
    """Unified training pipeline for all RL algorithms"""

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.env = WarehouseEnv(config_path)
        self.agent = None
        self.algorithm = self.config['training']['algorithm']

        # Training parameters
        self.total_timesteps = self.config['training']['total_timesteps']

        # Results storage
        self.training_history = {
            'rewards': [],
            'episode_lengths': [],
            'metrics': {}
        }

    def setup_agent(self, algorithm: Optional[str] = None):
        """Initialize RL agent"""
        if algorithm:
            self.algorithm = algorithm

        # Get observation and action dimensions
        obs_dim = self.env.observation_space.shape[0]
        action_dims = [5, 5, 2, 2]  # From MultiDiscrete action space

        # Create agent based on algorithm
        if self.algorithm == "PPO":
            self.agent = PPOAgent(self.config, obs_dim, action_dims)
        elif self.algorithm == "DQN":
            self.agent = DQNAgent(self.config, obs_dim, action_dims)
        elif self.algorithm == "A3C":
            self.agent = A3CAgent(self.config, obs_dim, action_dims)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        print(f"Initialized {self.algorithm} agent")

    def train(self, save_path: Optional[str] = None, total_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """Train the agent"""
        if self.agent is None:
            self.setup_agent()
        
        timesteps = total_timesteps if total_timesteps is not None else self.total_timesteps
        print(f"Starting training for {timesteps} timesteps...")

        # Train agent
        results = self.agent.train(self.env, timesteps)

        # Store results
        self.training_history = results

        # Save model if path provided
        if save_path:
            self.save_model(save_path)

        print("Training complete!")
        return results

    def save_model(self, path: str):
        """Save trained model and training history"""
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save agent model
        model_path = path if path.endswith('.pth') else f"{path}.pth"
        self.agent.save(model_path)

        # Save training history
        history_path = model_path.replace('.pth', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)

        # Save config
        config_path = model_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'algorithm': self.algorithm,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        print(f"Model saved to {model_path}")

    def load_model(self, path: str):
        """Load trained model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        # Load config to determine algorithm
        config_path = path.replace('.pth', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                self.algorithm = saved_config['algorithm']

        # Setup agent
        self.setup_agent()

        # Load model weights
        self.agent.load(path)

        # Load training history if available
        history_path = path.replace('.pth', '_history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)

        print(f"Model loaded from {path}")

    def evaluate(self, num_episodes: int = 100, render: bool = False) -> Dict[str, Any]:
        """Evaluate trained agent"""
        if self.agent is None:
            raise ValueError("No agent loaded. Train or load a model first.")

        episode_rewards = []
        episode_lengths = []
        completion_rates = []
        total_distances = []
        collision_counts = []

        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < 1000:
                # Get actions from agent
                actions = {}
                for agent_id in range(self.env.num_lgvs):
                    action, _ = self.agent.predict(observations[agent_id], deterministic=True)
                    actions[agent_id] = action

                # Step environment
                observations, rewards, dones, truncated, info = self.env.step(actions)

                episode_reward += sum(rewards.values())
                episode_length += 1

                done = dones.get('__all__', False)

            # Store metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            completion_rates.append(info['completion_rate'])
            total_distances.append(info['total_distance'])
            collision_counts.append(info['total_collisions'])

        # Compute statistics
        results = {
            'algorithm': self.algorithm,
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_completion_rate': np.mean(completion_rates),
            'mean_distance': np.mean(total_distances),
            'mean_collisions': np.mean(collision_counts),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'completion_rates': completion_rates,
            'total_distances': total_distances,
            'collision_counts': collision_counts
        }

        return results

    def get_training_curves(self) -> Dict[str, list]:
        """Get training curves for plotting"""
        if not self.training_history:
            return {}

        return {
            'rewards': self.training_history.get('rewards', []),
            'episode_lengths': self.training_history.get('episode_lengths', []),
            'metrics': self.training_history.get('training_metrics', {})
        }
