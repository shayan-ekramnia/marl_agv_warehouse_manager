"""
Quick training test with improved system
Tests PPO for a small number of timesteps to verify learning
"""
import numpy as np
from src.training.trainer import Trainer

def main():
    print("="*70)
    print("QUICK TRAINING TEST - PPO (10k timesteps)")
    print("="*70)

    # Create trainer
    trainer = Trainer('config.yaml')

    print("\nEnvironment Setup:")
    print(f"  Grid: {trainer.env.width}x{trainer.env.height}")
    print(f"  LGVs: {trainer.env.num_lgvs}")
    print(f"  Pallets: {trainer.env.num_pallets}")
    print(f"  Max Steps: {trainer.env.max_steps}")

    print("\nReward Configuration:")
    for key, value in trainer.env.rewards.items():
        print(f"  {key}: {value}")

    # Setup PPO agent
    print("\n" + "-"*70)
    print("Setting up PPO agent...")
    trainer.setup_agent('PPO')

    print(f"  Observation dim: {trainer.agent.obs_dim}")
    print(f"  Action dims: {trainer.agent.action_dims}")
    print(f"  Learning rate: {trainer.agent.lr}")
    print(f"  Batch size: {trainer.agent.batch_size}")
    print(f"  N-steps: {trainer.agent.n_steps}")

    # Quick training run
    print("\n" + "-"*70)
    print("Starting training (10,000 timesteps)...")
    print("This should take ~2-3 minutes")
    print("-"*70)

    results = trainer.train(
        total_timesteps=10000,
        save_path='models/PPO_quick_test.pth'
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    print("\nTraining Results:")
    print(f"  Episodes: {results['num_episodes']}")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")
    print(f"  Mean Completion Rate: {results['mean_completion_rate']:.1%}")
    print(f"  Mean Distance: {results['mean_distance']:.1f}")
    print(f"  Mean Collisions: {results['mean_collisions']:.1f}")

    # Analyze learning
    print("\n" + "-"*70)
    print("Learning Analysis:")

    rewards = results['rewards']
    if len(rewards) >= 5:
        early_avg = np.mean(rewards[:5])
        late_avg = np.mean(rewards[-5:])
        improvement = late_avg - early_avg

        print(f"  First 5 episodes avg: {early_avg:.2f}")
        print(f"  Last 5 episodes avg: {late_avg:.2f}")
        print(f"  Improvement: {improvement:+.2f}")

        if improvement > 10:
            print("  ✅ LEARNING DETECTED - Agent is improving!")
        elif improvement > 0:
            print("  ⚠️  SLIGHT IMPROVEMENT - May need more training")
        else:
            print("  ❌ NO IMPROVEMENT - Check hyperparameters")

    # Check completion rates
    if 'completion_rates' in results:
        completion = results['completion_rates']
        if len(completion) >= 5:
            early_comp = np.mean(completion[:5])
            late_comp = np.mean(completion[-5:])

            print(f"  First 5 episodes completion: {early_comp:.1%}")
            print(f"  Last 5 episodes completion: {late_comp:.1%}")

            if late_comp > early_comp:
                print("  ✅ Task completion improving!")

    print("\n" + "="*70)
    print("RECOMMENDATION:")

    if results['mean_reward'] > 0:
        print("  ✅ Positive rewards - System working correctly")
        print("  💡 Run full training: python run_training.py --algorithm PPO --timesteps 100000")
    elif results['mean_reward'] > -100:
        print("  ⚠️  Small negative rewards - Needs more training")
        print("  💡 Try: python run_training.py --algorithm PPO --timesteps 50000")
    else:
        print("  ❌ Large negative rewards - Check configuration")
        print("  💡 Review IMPROVEMENTS_V2.md for troubleshooting")

    print("="*70)

if __name__ == "__main__":
    main()
