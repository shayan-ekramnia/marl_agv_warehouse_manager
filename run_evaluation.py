"""
Command-line evaluation script
Usage: python run_evaluation.py --model models/PPO_model.pth --episodes 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.environment.warehouse_env import WarehouseEnv
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.baselines.baseline_runner import BaselineRunner


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model (for RL agents)")
    parser.add_argument("--baseline", type=str, default=None,
                       choices=["A_star", "Dijkstra", "Greedy", "Random"],
                       help="Baseline algorithm to evaluate")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Config file path")

    args = parser.parse_args()

    # Initialize environment
    env = WarehouseEnv(args.config)
    evaluator = Evaluator(env)

    if args.model:
        # Evaluate RL model
        print(f"Evaluating RL model: {args.model}")

        trainer = Trainer(args.config)
        trainer.load_model(args.model)

        results = evaluator.evaluate_rl_agent(trainer, args.episodes)

        print("\n" + "="*50)
        print(f"RL MODEL EVALUATION - {trainer.algorithm}")
        print("="*50)

    elif args.baseline:
        # Evaluate baseline
        print(f"Evaluating baseline: {args.baseline}")

        results = evaluator.evaluate_baseline(args.baseline, args.episodes)

        print("\n" + "="*50)
        print(f"BASELINE EVALUATION - {args.baseline}")
        print("="*50)

    else:
        print("Error: Must specify either --model or --baseline")
        return

    # Print results
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results.get('std_reward', 0):.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.0f}")
    print(f"Completion Rate: {results.get('mean_completion_rate', 0):.1%}")
    print(f"Mean Distance: {results.get('mean_distance', 0):.2f}")
    print(f"Mean Collisions: {results.get('mean_collisions', 0):.2f}")
    print(f"Efficiency Score: {results.get('efficiency_score', 0):.1f}/100")
    print("="*50)


if __name__ == "__main__":
    main()
