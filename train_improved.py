"""
Improved training script with better monitoring and early stopping
"""
import sys
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.environment.warehouse_env import WarehouseEnv

print("="*70)
print("IMPROVED RL TRAINING")
print("="*70)

# Load improved config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"\nConfiguration:")
print(f"  Warehouse: {config['warehouse']['width']}x{config['warehouse']['height']}")
print(f"  LGVs: {config['warehouse']['num_lgvs']}")
print(f"  Pallets: {config['warehouse']['num_pallets']}")
print(f"  Shelves: {config['warehouse']['num_shelves']}")
print(f"\nReward Function:")
print(f"  Delivery success: +{config['rewards']['delivery_success']}")
print(f"  Progress reward: +{config['rewards']['progress_reward']} per unit")
print(f"  Distance penalty: {config['rewards']['distance_penalty']} per unit")
print(f"  Collision penalty: {config['rewards']['collision_penalty']}")
print(f"  Step penalty: {config['rewards']['step_penalty']}")

# Train all three algorithms
algorithms = ['PPO', 'DQN', 'A3C']
results = {}

for algo in algorithms:
    print(f"\n{'='*70}")
    print(f"Training {algo}")
    print(f"{'='*70}\n")

    trainer = Trainer('config.yaml')
    trainer.setup_agent(algo)

    # Train with reduced timesteps for testing
    train_results = trainer.train(save_path=f'models/{algo}_improved.pth')

    print(f"\n{algo} Training Complete:")
    print(f"  Mean Reward: {train_results['mean_reward']:.2f}")
    print(f"  Episodes: {len(train_results.get('rewards', []))}")

    # Quick evaluation
    print(f"\nEvaluating {algo}...")
    eval_results = trainer.evaluate(num_episodes=10)

    print(f"{algo} Evaluation Results:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"  Completion Rate: {eval_results['mean_completion_rate']:.1%}")
    print(f"  Mean Distance: {eval_results['mean_distance']:.2f}")
    print(f"  Mean Collisions: {eval_results['mean_collisions']:.2f}")

    results[algo] = eval_results

# Compare results
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}\n")

print(f"{'Algorithm':<12} {'Mean Reward':<15} {'Completion':<12} {'Collisions':<12}")
print("-"*60)

for algo, res in results.items():
    print(f"{algo:<12} {res['mean_reward']:<15.2f} {res['mean_completion_rate']:<12.1%} {res['mean_collisions']:<12.2f}")

print(f"\n{'='*70}")

# Find best algorithm
best_algo = max(results.items(), key=lambda x: x[1]['mean_reward'])[0]
print(f"Best Algorithm: {best_algo}")
print(f"Best Reward: {results[best_algo]['mean_reward']:.2f}")
print(f"{'='*70}\n")

print("✅ Training and evaluation complete!")
print("Models saved to models/ directory")
