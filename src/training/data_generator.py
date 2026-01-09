"""
Synthetic data generation for RL training
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os

from ..environment.warehouse_env import WarehouseEnv


class DataGenerator:
    """Generate synthetic warehouse operation data"""

    def __init__(self, env: WarehouseEnv):
        self.env = env

    def generate_random_episodes(self, num_episodes: int = 1000) -> pd.DataFrame:
        """
        Generate random episodes for exploration

        Returns:
            DataFrame with episode statistics
        """
        data = []

        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            episode_data = {
                'episode': episode,
                'total_reward': 0,
                'episode_length': 0,
                'deliveries': 0,
                'collisions': 0,
                'total_distance': 0
            }

            done = False
            step = 0

            while not done and step < 1000:
                # Random actions
                actions = {
                    i: self.env.action_space.sample()
                    for i in range(self.env.num_lgvs)
                }

                observations, rewards, dones, truncated, info = self.env.step(actions)

                episode_data['total_reward'] += sum(rewards.values())
                step += 1

                done = dones.get('__all__', False)

            # Store episode info
            episode_data['episode_length'] = step
            episode_data['deliveries'] = info['total_deliveries']
            episode_data['collisions'] = info['total_collisions']
            episode_data['total_distance'] = info['total_distance']
            episode_data['completion_rate'] = info['completion_rate']

            data.append(episode_data)

        df = pd.DataFrame(data)
        return df

    def generate_state_action_pairs(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate state-action pairs for offline learning

        Returns:
            states, actions arrays
        """
        states = []
        actions = []

        observations, _ = self.env.reset()
        samples_collected = 0

        while samples_collected < num_samples:
            # Random actions
            random_actions = {
                i: self.env.action_space.sample()
                for i in range(self.env.num_lgvs)
            }

            # Store state-action pairs
            for agent_id in range(self.env.num_lgvs):
                states.append(observations[agent_id])
                actions.append(random_actions[agent_id])
                samples_collected += 1

                if samples_collected >= num_samples:
                    break

            # Step environment
            observations, rewards, dones, truncated, info = self.env.step(random_actions)

            # Reset if done
            if dones.get('__all__', False):
                observations, _ = self.env.reset()

        return np.array(states), np.array(actions)

    def generate_expert_demonstrations(self, planner_type: str = 'A_star', num_episodes: int = 100) -> Dict:
        """
        Generate demonstrations using classical planners (for imitation learning)

        Args:
            planner_type: Type of planner to use ('A_star', 'Dijkstra', etc.)
            num_episodes: Number of demonstration episodes

        Returns:
            Dictionary with demonstrations
        """
        from ..baselines import BaselineRunner

        runner = BaselineRunner(self.env)
        results = runner.run_algorithm(planner_type, num_episodes)

        return {
            'algorithm': planner_type,
            'num_episodes': num_episodes,
            'performance': results
        }

    def save_dataset(self, data: pd.DataFrame, path: str):
        """Save dataset to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data.to_csv(path, index=False)
        print(f"Dataset saved to {path}")

    def load_dataset(self, path: str) -> pd.DataFrame:
        """Load dataset from file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")

        return pd.read_csv(path)

    def generate_scenario_dataset(self, scenarios: List[Dict], samples_per_scenario: int = 100) -> pd.DataFrame:
        """
        Generate dataset with specific scenarios

        Args:
            scenarios: List of scenario configurations
            samples_per_scenario: Number of samples per scenario

        Returns:
            DataFrame with scenario data
        """
        all_data = []

        for scenario_id, scenario in enumerate(scenarios):
            # Update environment configuration
            self._apply_scenario(scenario)

            # Generate samples
            for sample in range(samples_per_scenario):
                observations, _ = self.env.reset()

                sample_data = {
                    'scenario_id': scenario_id,
                    'sample_id': sample,
                    'num_lgvs': scenario.get('num_lgvs', self.env.num_lgvs),
                    'num_pallets': scenario.get('num_pallets', self.env.num_pallets),
                    'warehouse_size': f"{self.env.width}x{self.env.height}"
                }

                # Run episode
                done = False
                total_reward = 0
                steps = 0

                while not done and steps < 500:
                    actions = {i: self.env.action_space.sample() for i in range(self.env.num_lgvs)}
                    observations, rewards, dones, truncated, info = self.env.step(actions)

                    total_reward += sum(rewards.values())
                    steps += 1
                    done = dones.get('__all__', False)

                sample_data.update({
                    'total_reward': total_reward,
                    'steps': steps,
                    'completion_rate': info['completion_rate']
                })

                all_data.append(sample_data)

        return pd.DataFrame(all_data)

    def _apply_scenario(self, scenario: Dict):
        """Apply scenario configuration to environment"""
        # This would modify environment parameters
        # For now, just reset with current config
        pass

    def get_statistics(self, data: pd.DataFrame) -> Dict:
        """Compute statistics from dataset"""
        stats = {
            'num_episodes': len(data),
            'mean_reward': data['total_reward'].mean(),
            'std_reward': data['total_reward'].std(),
            'mean_length': data['episode_length'].mean(),
            'mean_completion_rate': data['completion_rate'].mean() if 'completion_rate' in data.columns else 0,
        }

        return stats
