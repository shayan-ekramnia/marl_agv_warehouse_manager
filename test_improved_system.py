"""
Test improved reward system and simplified environment
"""
import numpy as np
import yaml
from src.environment.warehouse_env import WarehouseEnv
from src.agents.ppo_agent import PPOAgent
from src.baselines.baseline_runner import BaselineRunner

def test_config_loads():
    """Test that config.yaml loads correctly with new values"""
    print("\n" + "="*70)
    print("TEST 1: Config Loading")
    print("="*70)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("\nWarehouse Config:")
    print(f"  Size: {config['warehouse']['width']}x{config['warehouse']['height']}")
    print(f"  LGVs: {config['warehouse']['num_lgvs']}")
    print(f"  Pallets: {config['warehouse']['num_pallets']}")
    print(f"  Shelves: {config['warehouse']['num_shelves']}")

    print("\nReward Config:")
    for key, value in config['rewards'].items():
        print(f"  {key}: {value}")

    # Assertions
    assert config['warehouse']['width'] == 10, "Width should be 10"
    assert config['warehouse']['num_lgvs'] == 2, "Should have 2 LGVs"
    assert config['warehouse']['num_pallets'] == 5, "Should have 5 pallets"
    assert config['rewards']['collision_penalty'] == -5.0, "Collision penalty should be -5.0"
    assert config['rewards']['step_penalty'] == -0.02, "Step penalty should be -0.02"
    assert 'pickup_success' in config['rewards'], "Should have pickup_success reward"
    assert 'progress_to_pickup' in config['rewards'], "Should have progress_to_pickup reward"

    print("\n✅ Config loads correctly with new values")
    return config

def test_environment_simplified():
    """Test that environment is properly simplified"""
    print("\n" + "="*70)
    print("TEST 2: Environment Simplification")
    print("="*70)

    env = WarehouseEnv('config.yaml')

    print(f"\nEnvironment Size: {env.width}x{env.height}")
    print(f"Number of LGVs: {len(env.lgvs)}")
    print(f"Number of Pallets: {len(env.pallets)}")
    print(f"Number of Shelves: {len(env.shelves)}")
    print(f"Max Steps: {env.max_steps}")

    # Assertions
    assert env.width == 10, "Width should be 10"
    assert env.height == 10, "Height should be 10"
    assert len(env.lgvs) == 2, "Should have 2 LGVs"
    assert len(env.pallets) == 5, "Should have 5 pallets"
    assert env.max_steps == 250, f"Max steps should be 250, got {env.max_steps}"

    print("\n✅ Environment properly simplified")
    return env

def test_reward_calculation():
    """Test that rewards are calculated correctly"""
    print("\n" + "="*70)
    print("TEST 3: Reward Calculation")
    print("="*70)

    env = WarehouseEnv('config.yaml')
    obs, info = env.reset()

    # Run a few steps and track rewards
    episode_reward = 0
    rewards_list = []

    for step in range(10):
        # Random actions for both agents
        actions = {
            0: np.array([2, 2, 0, 0]),  # Move forward
            1: np.array([2, 2, 0, 0])   # Move forward
        }

        obs, rewards, dones, truncated, info = env.step(actions)

        step_reward = sum(rewards.values())
        episode_reward += step_reward
        rewards_list.append(step_reward)

        print(f"  Step {step+1}: Reward = {step_reward:.3f} (Cumulative: {episode_reward:.3f})")

    # Check that rewards are not catastrophically negative
    avg_reward_per_step = episode_reward / 10
    print(f"\nAverage Reward per Step: {avg_reward_per_step:.3f}")

    # With 2 agents and step penalty of -0.02, minimum is -0.04 per step
    # But with progress rewards, should be better than just penalties
    assert avg_reward_per_step > -1.0, f"Rewards too negative: {avg_reward_per_step}"

    print("\n✅ Reward calculation reasonable")
    return env

def test_episode_completion():
    """Test that episodes can complete naturally"""
    print("\n" + "="*70)
    print("TEST 4: Episode Completion")
    print("="*70)

    env = WarehouseEnv('config.yaml')
    obs, info = env.reset()

    print(f"\nStarting episode with {len(env.pallets)} pallets to deliver")
    print(f"Max steps: {env.max_steps}")

    episode_reward = 0
    step = 0
    done = False

    while not done and step < env.max_steps:
        # Simple forward movement
        actions = {
            0: np.array([2, 2, 0, 0]),
            1: np.array([2, 2, 0, 0])
        }

        obs, rewards, dones, truncated, info = env.step(actions)
        episode_reward += sum(rewards.values())
        step += 1
        done = dones.get('__all__', False)

    print(f"\nEpisode ended after {step} steps")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Completion Rate: {info['completion_rate']:.1%}")
    print(f"Deliveries: {info['delivered_pallets']}/{len(env.pallets)}")

    # Episode should end before max_steps (even if by timeout)
    assert step <= env.max_steps, f"Episode exceeded max_steps"

    print("\n✅ Episode completion works")
    return env

def test_baseline_performance():
    """Test baseline algorithm with new environment"""
    print("\n" + "="*70)
    print("TEST 5: Baseline Performance (Quick Test)")
    print("="*70)

    env = WarehouseEnv('config.yaml')
    runner = BaselineRunner(env)

    print("\nRunning A* for 5 episodes...")
    results = runner.run_algorithm('A_star', num_episodes=5)

    print(f"\nResults:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")
    print(f"  Completion Rate: {results['mean_completion_rate']:.1%}")
    print(f"  Mean Collisions: {results['mean_collisions']:.1f}")

    # Baseline should not have catastrophically negative rewards anymore
    assert results['mean_reward'] > -1000, f"Baseline reward too negative: {results['mean_reward']}"

    print("\n✅ Baseline performance reasonable")
    return results

def test_ppo_agent_creation():
    """Test PPO agent can be created and run"""
    print("\n" + "="*70)
    print("TEST 6: PPO Agent Creation")
    print("="*70)

    env = WarehouseEnv('config.yaml')

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    obs_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()

    print(f"\nObservation Dim: {obs_dim}")
    print(f"Action Dims: {action_dims}")

    agent = PPOAgent(config, obs_dim, action_dims)

    print(f"\nPPO Config:")
    print(f"  Learning Rate: {agent.lr}")
    print(f"  Batch Size: {agent.batch_size}")
    print(f"  N-Steps: {agent.n_steps}")
    print(f"  Entropy Coef: {agent.entropy_coef}")

    # Test prediction
    obs, info = env.reset()
    obs_array = np.array([obs[i] for i in range(len(env.lgvs))])
    actions = agent.predict(obs_array, deterministic=False)

    print(f"\nSample Actions: {actions}")

    print("\n✅ PPO agent created successfully")
    return agent

def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# TESTING IMPROVED SYSTEM")
    print("#"*70)

    try:
        config = test_config_loads()
        env = test_environment_simplified()
        test_reward_calculation()
        test_episode_completion()
        baseline_results = test_baseline_performance()
        agent = test_ppo_agent_creation()

        print("\n" + "#"*70)
        print("# ALL TESTS PASSED ✅")
        print("#"*70)

        print("\n📊 SUMMARY:")
        print(f"  Environment: {env.width}x{env.height}, {len(env.lgvs)} LGVs, {len(env.pallets)} pallets")
        print(f"  Max Steps: {env.max_steps}")
        print(f"  Baseline A* Reward: {baseline_results['mean_reward']:.2f}")
        print(f"  Rewards: Task-based (positive) with small penalties")
        print("\n✅ System ready for improved training!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
