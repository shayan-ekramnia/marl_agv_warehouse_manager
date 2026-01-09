"""
Evaluation metrics for MARL system
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy import stats


class MetricsCalculator:
    """Calculate performance metrics"""

    @staticmethod
    def calculate_episode_metrics(episode_data: Dict) -> Dict[str, float]:
        """Calculate metrics for a single episode"""
        metrics = {
            'total_reward': episode_data.get('total_reward', 0),
            'episode_length': episode_data.get('episode_length', 0),
            'completion_rate': episode_data.get('completion_rate', 0),
            'total_distance': episode_data.get('total_distance', 0),
            'collision_count': episode_data.get('total_collisions', 0),
            'avg_delivery_time': episode_data.get('avg_delivery_time', 0),
        }

        # Derived metrics
        if metrics['episode_length'] > 0:
            metrics['avg_reward_per_step'] = metrics['total_reward'] / metrics['episode_length']
            metrics['distance_per_step'] = metrics['total_distance'] / metrics['episode_length']
            metrics['collision_rate'] = metrics['collision_count'] / metrics['episode_length']
        else:
            metrics['avg_reward_per_step'] = 0
            metrics['distance_per_step'] = 0
            metrics['collision_rate'] = 0

        # Efficiency metrics
        if metrics['total_distance'] > 0:
            metrics['deliveries_per_distance'] = episode_data.get('total_deliveries', 0) / metrics['total_distance']
        else:
            metrics['deliveries_per_distance'] = 0

        return metrics

    @staticmethod
    def aggregate_metrics(episodes: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics across multiple episodes"""
        if not episodes:
            return {}

        # Extract all metrics
        all_metrics = {}
        for episode in episodes:
            for key, value in episode.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # Compute statistics
        aggregated = {}
        for key, values in all_metrics.items():
            if isinstance(values[0], (int, float)):
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
                aggregated[f'{key}_median'] = np.median(values)

        return aggregated

    @staticmethod
    def calculate_efficiency_score(metrics: Dict) -> float:
        """
        Calculate overall efficiency score (0-100)

        Considers:
        - Completion rate (40%)
        - Collision avoidance (20%)
        - Distance efficiency (20%)
        - Time efficiency (20%)
        """
        completion_score = metrics.get('completion_rate', 0) * 40

        collision_rate = metrics.get('collision_rate', 1)
        collision_score = max(0, 1 - collision_rate) * 20

        # Distance efficiency (lower is better, normalized)
        distance_per_step = metrics.get('distance_per_step', 1)
        distance_score = max(0, min(1, 1 / (1 + distance_per_step))) * 20

        # Time efficiency (shorter episodes are better for same completion)
        episode_length = metrics.get('episode_length', 1000)
        time_score = max(0, min(1, 500 / episode_length)) * 20

        total_score = completion_score + collision_score + distance_score + time_score
        return total_score


class ComparisonAnalyzer:
    """Compare multiple algorithms"""

    def __init__(self):
        self.results = {}

    def add_results(self, algorithm_name: str, results: Dict):
        """Add results for an algorithm"""
        self.results[algorithm_name] = results

    def compare_algorithms(self) -> pd.DataFrame:
        """Create comparison table"""
        comparison_data = []

        for algo_name, results in self.results.items():
            row = {'Algorithm': algo_name}

            # Key metrics
            metrics = [
                'mean_reward',
                'mean_episode_length',
                'mean_completion_rate',
                'mean_distance',
                'mean_collisions'
            ]

            for metric in metrics:
                value = results.get(metric, 0)
                row[metric] = value

            # Calculate efficiency score
            efficiency = MetricsCalculator.calculate_efficiency_score(results)
            row['efficiency_score'] = efficiency

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Rank algorithms
        df['rank'] = df['efficiency_score'].rank(ascending=False)

        return df.sort_values('rank')

    def statistical_significance_test(self, algo1: str, algo2: str, metric: str = 'mean_reward') -> Dict:
        """Test statistical significance between two algorithms"""
        if algo1 not in self.results or algo2 not in self.results:
            return {'error': 'Algorithm not found'}

        # Get episode data
        data1 = self.results[algo1].get('episode_rewards', [])
        data2 = self.results[algo2].get('episode_rewards', [])

        if not data1 or not data2:
            return {'error': 'No episode data available'}

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)

        # Effect size (Cohen's d)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1), np.std(data2)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        return {
            'algorithm_1': algo1,
            'algorithm_2': algo2,
            'mean_1': mean1,
            'mean_2': mean2,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_effect_size(abs(cohens_d))
        }

    @staticmethod
    def _interpret_effect_size(d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'

    def best_algorithm(self, metric: str = 'efficiency_score') -> str:
        """Get best performing algorithm"""
        comparison = self.compare_algorithms()
        if metric == 'efficiency_score':
            best = comparison.iloc[0]['Algorithm']
        else:
            best = comparison.loc[comparison[metric].idxmax(), 'Algorithm']

        return best

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for all algorithms"""
        summary = {}

        for algo_name, results in self.results.items():
            summary[algo_name] = {
                'num_episodes': results.get('num_episodes', 0),
                'mean_reward': results.get('mean_reward', 0),
                'std_reward': results.get('std_reward', 0),
                'completion_rate': results.get('mean_completion_rate', 0),
                'efficiency_score': MetricsCalculator.calculate_efficiency_score(results)
            }

        return summary
