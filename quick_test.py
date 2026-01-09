"""
Quick test to verify all components work end-to-end
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("QUICK FUNCTIONALITY TEST")
print("="*70)

# Test 1: Small training run
print("\n[1/3] Testing short training run...")
try:
    from src.training.trainer import Trainer

    # Create a trainer with reduced timesteps
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Drastically reduce for quick test
    config['training']['total_timesteps'] = 5000

    # Save temp config
    with open('config_test.yaml', 'w') as f:
        yaml.dump(config, f)

    trainer = Trainer('config_test.yaml')
    trainer.setup_agent('PPO')

    # Train for very short time
    results = trainer.train(save_path='models/test_ppo.pth')

    # Check results have proper structure
    assert 'rewards' in results, "Missing rewards list"
    assert 'episode_lengths' in results, "Missing episode_lengths list"
    assert len(results['rewards']) > 0, "No episodes completed"

    print(f"✅ Training successful!")
    print(f"   - Episodes: {len(results['rewards'])}")
    print(f"   - Mean reward: {results['mean_reward']:.2f}")

except Exception as e:
    print(f"❌ Training test failed: {e}")
    sys.exit(1)

# Test 2: Evaluation
print("\n[2/3] Testing evaluation...")
try:
    results = trainer.evaluate(num_episodes=5)

    # Check all required fields
    required_fields = ['episode_rewards', 'episode_lengths', 'completion_rates',
                      'total_distances', 'collision_counts']
    for field in required_fields:
        assert field in results, f"Missing field: {field}"
        assert len(results[field]) == 5, f"Wrong length for {field}"

    print(f"✅ Evaluation successful!")
    print(f"   - Mean reward: {results['mean_reward']:.2f}")
    print(f"   - Completion rate: {results['mean_completion_rate']:.1%}")

except Exception as e:
    print(f"❌ Evaluation test failed: {e}")
    sys.exit(1)

# Test 3: Baseline
print("\n[3/3] Testing baseline algorithms...")
try:
    from src.baselines.baseline_runner import BaselineRunner
    from src.environment.warehouse_env import WarehouseEnv

    env = WarehouseEnv('config.yaml')
    runner = BaselineRunner(env)

    # Run A* for a few episodes
    results = runner.run_algorithm('A_star', num_episodes=3)

    # Check results
    assert 'mean_reward' in results
    assert 'episode_rewards' in results
    assert len(results['episode_rewards']) == 3

    print(f"✅ Baseline test successful!")
    print(f"   - Mean reward: {results['mean_reward']:.2f}")
    print(f"   - Episodes: {len(results['episode_rewards'])}")

except Exception as e:
    print(f"❌ Baseline test failed: {e}")
    sys.exit(1)

# Test 4: Visualization
print("\n[4/4] Testing visualization...")
try:
    from src.visualization.plotter import Plotter

    # Test training curve plot
    training_data = {
        'rewards': results['episode_rewards'],
        'episode_lengths': results['episode_lengths']
    }

    fig = Plotter.plot_training_curves(training_data, 'Test')
    assert fig is not None, "Failed to create training curves"

    # Test warehouse layout plot
    env_state = env.get_state()
    fig2 = Plotter.plot_warehouse_layout(env_state)
    assert fig2 is not None, "Failed to create warehouse layout"

    print(f"✅ Visualization test successful!")

except Exception as e:
    print(f"❌ Visualization test failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("ALL QUICK TESTS PASSED! ✅")
print("="*70)
print("\nSystem is fully functional with real data!")
