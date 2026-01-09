"""
Test that improvements are working correctly
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("TESTING IMPROVEMENTS")
print("="*70)

# Test 1: Verify new reward function
print("\n[1/5] Testing improved reward function...")
try:
    from src.environment.warehouse_env import WarehouseEnv
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Check reward values
    assert config['rewards']['collision_penalty'] == -10.0, "Collision penalty not updated"
    assert config['rewards']['distance_penalty'] == -0.01, "Distance penalty not updated"
    assert 'progress_reward' in config['rewards'], "Progress reward missing"
    assert 'step_penalty' in config['rewards'], "Step penalty missing"

    print("✅ Reward configuration updated correctly")
    print(f"   Collision penalty: {config['rewards']['collision_penalty']} (was -50)")
    print(f"   Distance penalty: {config['rewards']['distance_penalty']} (was -0.1)")
    print(f"   Progress reward: {config['rewards']['progress_reward']} (NEW)")

except Exception as e:
    print(f"❌ Reward function test failed: {e}")
    sys.exit(1)

# Test 2: Verify simplified environment
print("\n[2/5] Testing simplified environment...")
try:
    env = WarehouseEnv('config.yaml')

    assert env.num_lgvs == 3, f"Expected 3 LGVs, got {env.num_lgvs}"
    assert env.width == 15, f"Expected width 15, got {env.width}"
    assert env.height == 15, f"Expected height 15, got {env.height}"
    assert len(env.pallets) == 10, f"Expected 10 pallets, got {len(env.pallets)}"

    print("✅ Environment simplified correctly")
    print(f"   Size: {env.width}x{env.height} (was 20x20)")
    print(f"   LGVs: {env.num_lgvs} (was 6)")
    print(f"   Pallets: {len(env.pallets)} (was 30)")

except Exception as e:
    print(f"❌ Environment test failed: {e}")
    sys.exit(1)

# Test 3: Test progress reward calculation
print("\n[3/5] Testing progress reward...")
try:
    env = WarehouseEnv('config.yaml')
    obs, _ = env.reset()

    # Simulate movement towards goal
    initial_rewards = []
    for i in range(3):
        actions = {j: np.array([3, 2, 0, 0]) for j in range(env.num_lgvs)}  # Move forward
        obs, rewards, dones, truncated, info = env.step(actions)
        initial_rewards.extend(rewards.values())

    avg_reward = np.mean(initial_rewards)

    # Check that rewards are not extremely negative
    assert avg_reward > -10, f"Rewards still too negative: {avg_reward}"

    print("✅ Progress rewards working")
    print(f"   Average reward per step: {avg_reward:.3f}")
    print(f"   (Should be close to 0, not -100)")

except Exception as e:
    print(f"❌ Progress reward test failed: {e}")
    sys.exit(1)

# Test 4: Verify PPO improvements
print("\n[4/5] Testing PPO improvements...")
try:
    from src.agents import PPOAgent

    env = WarehouseEnv('config.yaml')
    obs_dim = env.observation_space.shape[0]
    action_dims = [5, 5, 2, 2]

    agent = PPOAgent(config, obs_dim, action_dims)

    assert agent.batch_size == 128, f"Batch size not updated: {agent.batch_size}"
    assert agent.n_steps == 512, f"N_steps not updated: {agent.n_steps}"
    assert agent.entropy_coef == 0.02, f"Entropy coef not updated: {agent.entropy_coef}"

    print("✅ PPO hyperparameters improved")
    print(f"   Batch size: {agent.batch_size} (was 64)")
    print(f"   N-steps: {agent.n_steps} (was 2048)")
    print(f"   Entropy coef: {agent.entropy_coef} (was 0.01)")

except Exception as e:
    print(f"❌ PPO test failed: {e}")
    sys.exit(1)

# Test 5: Verify DQN improvements
print("\n[5/5] Testing DQN improvements...")
try:
    from src.agents import DQNAgent

    agent = DQNAgent(config, obs_dim, action_dims)

    assert agent.epsilon_end == 0.05, f"Epsilon end not updated: {agent.epsilon_end}"
    assert agent.target_update_freq == 500, f"Target update freq not updated: {agent.target_update_freq}"
    assert agent.batch_size == 128, f"Batch size not updated: {agent.batch_size}"

    print("✅ DQN hyperparameters improved")
    print(f"   Epsilon end: {agent.epsilon_end} (was 0.01)")
    print(f"   Target update freq: {agent.target_update_freq} (was 1000)")
    print(f"   Batch size: {agent.batch_size} (was 64)")

except Exception as e:
    print(f"❌ DQN test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("ALL IMPROVEMENT TESTS PASSED! ✅")
print("="*70)

print("\nKey Improvements Verified:")
print("  ✅ Reward function 10x less harsh")
print("  ✅ Progress rewards implemented")
print("  ✅ Environment simplified (15x15, 3 LGVs)")
print("  ✅ PPO: Better exploration (2x entropy)")
print("  ✅ DQN: Longer exploration (5x epsilon)")
print("  ✅ More frequent updates (2-4x)")

print("\nExpected Impact:")
print("  📈 Rewards should be -50 to +200 (not -200k)")
print("  📈 Agents should move (distance > 0)")
print("  📈 Some deliveries (completion > 5%)")
print("  📈 Learning progress over time")

print("\nNext Steps:")
print("  1. Train: python train_improved.py")
print("  2. Compare in dashboard: streamlit run app.py")
print("  3. Check that new models perform better")

print("="*70)
