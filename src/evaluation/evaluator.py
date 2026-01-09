"""
Comprehensive evaluation system
"""
from typing import Dict, List, Any, Optional
import numpy as np

from ..environment.warehouse_env import WarehouseEnv
from ..training.trainer import Trainer
from ..baselines.baseline_runner import BaselineRunner
from .metrics import MetricsCalculator, ComparisonAnalyzer


class Evaluator:
    """Comprehensive evaluation of MARL algorithms"""

    def __init__(self, env: WarehouseEnv):
        self.env = env
        self.metrics_calculator = MetricsCalculator()
        self.comparison_analyzer = ComparisonAnalyzer()

    def evaluate_rl_agent(self, trainer: Trainer, num_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate RL agent"""
        results = trainer.evaluate(num_episodes=num_episodes)

        # Calculate additional metrics
        episode_metrics = []
        for i in range(len(results.get('episode_rewards', []))):
            episode_data = {
                'total_reward': results['episode_rewards'][i],
                'episode_length': results['episode_lengths'][i],
                'completion_rate': results.get('completion_rates', [0] * len(results['episode_rewards']))[i] if 'completion_rates' in results else 0,
                'total_distance': results.get('total_distances', [0] * len(results['episode_rewards']))[i] if 'total_distances' in results else 0,
                'total_collisions': results.get('collision_counts', [0] * len(results['episode_rewards']))[i] if 'collision_counts' in results else 0,
            }
            metrics = self.metrics_calculator.calculate_episode_metrics(episode_data)
            episode_metrics.append(metrics)

        # Aggregate metrics
        aggregated = self.metrics_calculator.aggregate_metrics(episode_metrics)
        results.update(aggregated)

        # Efficiency score
        results['efficiency_score'] = self.metrics_calculator.calculate_efficiency_score(results)

        return results

    def evaluate_baseline(self, algorithm: str, num_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate baseline algorithm"""
        runner = BaselineRunner(self.env)
        results = runner.run_algorithm(algorithm, num_episodes)

        # Add efficiency score
        results['efficiency_score'] = self.metrics_calculator.calculate_efficiency_score(results)

        return results

    def evaluate_all(self,
                     rl_trainers: Optional[Dict[str, Trainer]] = None,
                     baseline_algorithms: Optional[List[str]] = None,
                     num_episodes: int = 100) -> Dict[str, Dict]:
        """
        Evaluate all algorithms

        Args:
            rl_trainers: Dictionary of {name: Trainer} for RL algorithms
            baseline_algorithms: List of baseline algorithm names
            num_episodes: Number of evaluation episodes

        Returns:
            Dictionary with all results
        """
        all_results = {}

        # Evaluate RL agents
        if rl_trainers:
            for name, trainer in rl_trainers.items():
                print(f"Evaluating {name}...")
                results = self.evaluate_rl_agent(trainer, num_episodes)
                all_results[name] = results
                self.comparison_analyzer.add_results(name, results)

        # Evaluate baselines
        if baseline_algorithms:
            for algo in baseline_algorithms:
                print(f"Evaluating baseline: {algo}...")
                results = self.evaluate_baseline(algo, num_episodes)
                all_results[algo] = results
                self.comparison_analyzer.add_results(algo, results)

        return all_results

    def compare_with_baselines(self,
                               rl_trainer: Trainer,
                               baseline_algorithms: List[str] = ['A_star', 'Dijkstra'],
                               num_episodes: int = 100) -> Dict:
        """
        Compare RL agent with baseline algorithms

        Returns:
            Comparison results and analysis
        """
        # Evaluate RL agent
        rl_results = self.evaluate_rl_agent(rl_trainer, num_episodes)
        self.comparison_analyzer.add_results(rl_trainer.algorithm, rl_results)

        # Evaluate baselines
        baseline_results = {}
        for algo in baseline_algorithms:
            results = self.evaluate_baseline(algo, num_episodes)
            baseline_results[algo] = results
            self.comparison_analyzer.add_results(algo, results)

        # Create comparison
        comparison_table = self.comparison_analyzer.compare_algorithms()

        # Statistical tests
        statistical_tests = []
        for baseline in baseline_algorithms:
            test_result = self.comparison_analyzer.statistical_significance_test(
                rl_trainer.algorithm,
                baseline,
                'mean_reward'
            )
            statistical_tests.append(test_result)

        return {
            'rl_results': rl_results,
            'baseline_results': baseline_results,
            'comparison_table': comparison_table,
            'statistical_tests': statistical_tests,
            'best_algorithm': self.comparison_analyzer.best_algorithm()
        }

    def get_learning_curve_analysis(self, trainer: Trainer) -> Dict:
        """Analyze learning curves"""
        training_curves = trainer.get_training_curves()

        if not training_curves or not training_curves.get('rewards'):
            return {'error': 'No training data available'}

        rewards = training_curves['rewards']

        # Smoothed rewards (moving average)
        window = min(100, len(rewards) // 10)
        if window > 0:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        else:
            smoothed = rewards

        analysis = {
            'total_episodes': len(rewards),
            'initial_performance': np.mean(rewards[:10]) if len(rewards) >= 10 else 0,
            'final_performance': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
            'improvement': 0,
            'convergence_episode': self._find_convergence_point(smoothed),
            'stability': np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards),
            'smoothed_curve': smoothed.tolist() if isinstance(smoothed, np.ndarray) else smoothed
        }

        if analysis['initial_performance'] != 0:
            analysis['improvement'] = (analysis['final_performance'] - analysis['initial_performance']) / abs(analysis['initial_performance']) * 100

        return analysis

    @staticmethod
    def _find_convergence_point(rewards: np.ndarray, threshold: float = 0.01) -> int:
        """Find episode where learning converges"""
        if len(rewards) < 50:
            return len(rewards)

        # Find where variance stabilizes
        window = 50
        variances = []

        for i in range(len(rewards) - window):
            var = np.var(rewards[i:i + window])
            variances.append(var)

        if not variances:
            return len(rewards)

        # Find first point where variance stays below threshold
        mean_var = np.mean(variances)
        for i, var in enumerate(variances):
            if var < threshold * mean_var:
                # Check if it stays low
                if all(v < threshold * mean_var for v in variances[i:min(i + 20, len(variances))]):
                    return i

        return len(rewards)

    def generate_report(self, results: Dict) -> str:
        """Generate text report"""
        report = []
        report.append("=" * 80)
        report.append("EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        for algo_name, algo_results in results.items():
            report.append(f"\n{algo_name}")
            report.append("-" * 80)
            report.append(f"Mean Reward: {algo_results.get('mean_reward', 0):.2f} ± {algo_results.get('std_reward', 0):.2f}")
            report.append(f"Mean Episode Length: {algo_results.get('mean_episode_length', 0):.2f}")
            report.append(f"Completion Rate: {algo_results.get('mean_completion_rate', 0):.2%}")
            report.append(f"Mean Distance: {algo_results.get('mean_distance', 0):.2f}")
            report.append(f"Mean Collisions: {algo_results.get('mean_collisions', 0):.2f}")
            report.append(f"Efficiency Score: {algo_results.get('efficiency_score', 0):.2f}/100")

        report.append("\n" + "=" * 80)
        report.append("Best Algorithm: " + self.comparison_analyzer.best_algorithm())
        report.append("=" * 80)

        return "\n".join(report)
