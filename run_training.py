"""
Command-line training script
Usage: python run_training.py --algorithm PPO --timesteps 100000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train MARL agents")
    parser.add_argument("--algorithm", type=str, default="PPO",
                       choices=["PPO", "DQN", "A3C"],
                       help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Config file path")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save trained model")

    args = parser.parse_args()

    print(f"Training {args.algorithm} for {args.timesteps} timesteps...")

    # Initialize trainer
    trainer = Trainer(args.config)
    trainer.setup_agent(args.algorithm)

    # Train
    results = trainer.train(save_path=args.save_path or f"models/{args.algorithm}_model.pth")

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Mean Reward: {results.get('mean_reward', 0):.2f}")
    print(f"Mean Episode Length: {results.get('mean_length', 0):.0f}")
    print(f"Total Episodes: {len(results.get('rewards', []))}")
    print("="*50)


if __name__ == "__main__":
    main()
