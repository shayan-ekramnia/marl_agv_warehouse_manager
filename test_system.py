"""
System Integration Test
Tests all components to ensure they work together
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("MARL WAREHOUSE LGV OPTIMIZATION - SYSTEM TEST")
print("="*70)

# Test 1: Environment
print("\n[1/7] Testing Environment...")
try:
    from src.environment.warehouse_env import WarehouseEnv
    env = WarehouseEnv('config.yaml')
    observations, info = env.reset()
    print(f"✅ Environment initialized successfully")
    print(f"    - Number of LGVs: {env.num_lgvs}")
    print(f"    - Warehouse size: {env.width}x{env.height}")
    print(f"    - Number of pallets: {len(env.pallets)}")
except Exception as e:
    print(f"❌ Environment test failed: {e}")
    sys.exit(1)

# Test 2: Environment Step
print("\n[2/7] Testing Environment Step...")
try:
    actions = {i: env.action_space.sample() for i in range(env.num_lgvs)}
    observations, rewards, dones, truncated, info = env.step(actions)
    print(f"✅ Environment step successful")
    print(f"    - Reward received: {sum(rewards.values()):.2f}")
except Exception as e:
    print(f"❌ Environment step failed: {e}")
    sys.exit(1)

# Test 3: PPO Agent
print("\n[3/7] Testing PPO Agent...")
try:
    from src.agents import PPOAgent
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    obs_dim = env.observation_space.shape[0]
    action_dims = [5, 5, 2, 2]

    agent = PPOAgent(config, obs_dim, action_dims)
    action, _ = agent.predict(observations[0])
    print(f"✅ PPO Agent initialized and prediction successful")
    print(f"    - Action shape: {action.shape}")
except Exception as e:
    print(f"❌ PPO Agent test failed: {e}")
    sys.exit(1)

# Test 4: Baseline Algorithms
print("\n[4/7] Testing Baseline Algorithms...")
try:
    from src.baselines.pathfinding import AStarPlanner

    planner = AStarPlanner(env.grid)
    path = planner.find_path((5, 5), (10, 10))
    print(f"✅ A* Planner successful")
    if path:
        print(f"    - Path length: {len(path)}")
except Exception as e:
    print(f"❌ Baseline test failed: {e}")
    sys.exit(1)

# Test 5: Training Pipeline
print("\n[5/7] Testing Training Pipeline...")
try:
    from src.training.trainer import Trainer

    trainer = Trainer('config.yaml')
    trainer.setup_agent('PPO')
    print(f"✅ Training pipeline initialized")
    print(f"    - Algorithm: {trainer.algorithm}")
except Exception as e:
    print(f"❌ Training pipeline test failed: {e}")
    sys.exit(1)

# Test 6: Evaluation
print("\n[6/7] Testing Evaluation System...")
try:
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.metrics import MetricsCalculator

    evaluator = Evaluator(env)
    calculator = MetricsCalculator()

    episode_data = {
        'total_reward': 100,
        'episode_length': 50,
        'completion_rate': 0.8,
        'total_distance': 120,
        'total_collisions': 2
    }

    metrics = calculator.calculate_episode_metrics(episode_data)
    print(f"✅ Evaluation system working")
    print(f"    - Metrics calculated: {len(metrics)}")
except Exception as e:
    print(f"❌ Evaluation test failed: {e}")
    sys.exit(1)

# Test 7: Visualization
print("\n[7/7] Testing Visualization...")
try:
    from src.visualization.plotter import Plotter
    from src.visualization.animator import WarehouseAnimator

    env_state = env.get_state()
    fig = Plotter.plot_warehouse_layout(env_state)
    print(f"✅ Visualization system working")
    print(f"    - Plot generated successfully")

    animator = WarehouseAnimator(env.width, env.height)
    animator.add_frame(env_state)
    print(f"    - Animator initialized with {animator.get_frame_count()} frames")
except Exception as e:
    print(f"❌ Visualization test failed: {e}")
    sys.exit(1)

# Final Summary
print("\n" + "="*70)
print("ALL TESTS PASSED! ✅")
print("="*70)
print("\nSystem is ready to use!")
print("\nQuick Start:")
print("  1. Run Streamlit app:  streamlit run app.py")
print("  2. Or use run script:   ./run_app.sh")
print("  3. Or train via CLI:    python run_training.py --algorithm PPO")
print("="*70)
