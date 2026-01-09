"""
Streamlit Dashboard for MARL Warehouse LGV Optimization
Complete End-to-End Application
"""
import streamlit as st
import numpy as np
import pandas as pd
import yaml
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.environment.warehouse_env import WarehouseEnv
from src.agents import PPOAgent, DQNAgent, A3CAgent
from src.training.trainer import Trainer
from src.training.data_generator import DataGenerator
from src.baselines.baseline_runner import BaselineRunner
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import ComparisonAnalyzer, MetricsCalculator
from src.visualization.plotter import Plotter
from src.visualization.animator import WarehouseAnimator

# Page config
st.set_page_config(
    page_title="MARL Warehouse LGV Optimization",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []

def load_config():
    """Load configuration from yaml file"""
    default_config = {
        'warehouse': {
            'width': 20,
            'height': 20,
            'num_shelves': 15,
            'num_pallets': 30,
            'num_lgvs': 6
        },
        'lgv': {
            'max_speed': 2.0,
            'max_acceleration': 0.5,
            'turning_radius': 1.0,
            'load_capacity': 1,
            'loading_time': 5,
            'unloading_time': 5
        },
        'rewards': {
            'delivery_success': 100.0,
            'distance_penalty': -0.1,
            'collision_penalty': -50.0,
            'idle_penalty': -0.5,
            'efficiency_bonus': 10.0
        },
        'training': {
            'algorithm': 'PPO',
            'total_timesteps': 100000,
            'learning_rate': 0.0003,
            'batch_size': 64,
            'gamma': 0.99,
            'n_steps': 2048,
            'ent_coef': 0.01
        }
    }
    
    if os.path.exists('config.yaml'):
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                # Deep update default config with loaded config
                for key, value in config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                return default_config
        except Exception as e:
            st.error(f"Error loading config.yaml: {e}")
            return default_config
    return default_config

# Load config into session state if not present or force reload
if 'config' not in st.session_state:
    st.session_state.config = load_config()


def main():
    """Main application"""

    # Title
    st.markdown('<div class="main-header">🤖 MARL Warehouse LGV Optimization 🤖</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "⚙️ Configuration", "🎓 Training", "📊 Evaluation",
         "🎮 Simulation", "📈 Analysis", "📚 Research"]
    )

    # Route to pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "⚙️ Configuration":
        show_configuration_page()
    elif page == "🎓 Training":
        show_training_page()
    elif page == "📊 Evaluation":
        show_evaluation_page()
    elif page == "🎮 Simulation":
        show_simulation_page()
    elif page == "📈 Analysis":
        show_analysis_page()
    elif page == "📚 Research":
        show_research_page()


def show_home_page():
    """Home page with project overview"""
    st.header("Welcome to MARL Warehouse LGV Optimization System")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Project Objectives")
        st.write("""
        This project develops reinforcement learning algorithms to optimize
        automated guided vehicle (LGV) operations in warehouse environments.

        **Key Features:**
        - Multi-agent warehouse simulation
        - Multiple RL algorithms (PPO, DQN, A3C)
        - Baseline comparisons (A*, Dijkstra, Greedy)
        - Real-time visualization
        - Comprehensive evaluation metrics
        - Performance analysis tools
        """)

        st.subheader("🎯 Expected Outcomes")
        st.write("""
        - Trained RL models for path optimization
        - Performance comparison with baselines
        - Visualization of LGV movements
        - Understanding of optimal sequencing strategies
        - Complete analysis and insights
        """)

    with col2:
        st.subheader("🚀 Quick Start Guide")
        st.write("""
        1. **Configuration**: Set up warehouse parameters and RL settings
        2. **Training**: Train RL agents or run baseline algorithms
        3. **Evaluation**: Compare algorithm performance
        4. **Simulation**: Visualize LGV operations in real-time
        5. **Analysis**: Deep dive into results and metrics
        6. **Research**: Explore methodology and findings
        """)

        st.subheader("📊 System Status")

        # Check if environment is initialized
        env_status = "✅ Ready" if st.session_state.env else "⚠️ Not Initialized"
        st.write(f"**Environment:** {env_status}")

        # Check trained models
        num_models = len(st.session_state.trained_models)
        st.write(f"**Trained Models:** {num_models}")

        # Check evaluation results
        num_evals = len(st.session_state.evaluation_results)
        st.write(f"**Evaluation Results:** {num_evals}")

    st.markdown("---")

    # Project info
    st.subheader("📖 About This Project")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**University of Naples**\nRL Project 2026")

    with col2:
        st.info("**Algorithms**\nPPO, DQN, A3C, A*, Dijkstra")

    with col3:
        st.info("**Framework**\nPyTorch, Gymnasium, Streamlit")


def show_configuration_page():
    """Configuration page"""
    st.header("⚙️ System Configuration")

    tab1, tab2, tab3 = st.tabs(["Warehouse Setup", "RL Configuration", "Reward Design"])

    with tab1:
        st.subheader("Warehouse Environment Parameters")

        # Get defaults from config
        c_wh = st.session_state.config.get('warehouse', {})
        c_lgv = st.session_state.config.get('lgv', {})
        c_train = st.session_state.config.get('training', {})
        c_reward = st.session_state.config.get('rewards', {})

        col1, col2 = st.columns(2)

        with col1:
            warehouse_width = st.slider("Warehouse Width", 10, 50, c_wh.get('width', 20))
            warehouse_height = st.slider("Warehouse Height", 10, 50, c_wh.get('height', 20))
            num_shelves = st.slider("Number of Shelves", 5, 30, c_wh.get('num_shelves', 15))

        with col2:
            num_pallets = st.slider("Number of Pallets", 10, 100, c_wh.get('num_pallets', 30))
            num_lgvs = st.slider("Number of LGVs", 2, 10, c_wh.get('num_lgvs', 6))

        st.subheader("LGV Physical Constraints")

        col1, col2, col3 = st.columns(3)

        with col1:
            max_speed = st.number_input("Max Speed (m/s)", 0.5, 5.0, float(c_lgv.get('max_speed', 2.0)))
            max_accel = st.number_input("Max Acceleration (m/s²)", 0.1, 2.0, float(c_lgv.get('max_acceleration', 0.5)))

        with col2:
            turning_radius = st.number_input("Turning Radius (m)", 0.5, 3.0, float(c_lgv.get('turning_radius', 1.0)))
            load_capacity = st.number_input("Load Capacity", 1, 5, int(c_lgv.get('load_capacity', 1)))

        with col3:
            loading_time = st.number_input("Loading Time (s)", 1, 20, int(c_lgv.get('loading_time', 5)))
            unloading_time = st.number_input("Unloading Time (s)", 1, 20, int(c_lgv.get('unloading_time', 5)))

    with tab2:
        st.subheader("RL Training Configuration")

        col1, col2 = st.columns(2)

        algo_options = ["PPO", "DQN", "A3C"]
        current_algo = c_train.get('algorithm', 'PPO')
        algo_index = algo_options.index(current_algo) if current_algo in algo_options else 0

        with col1:
            algorithm = st.selectbox("Algorithm", algo_options, index=algo_index)
            total_timesteps = st.number_input("Total Timesteps", 10000, 1000000, int(c_train.get('total_timesteps', 100000)), 10000)
            learning_rate = st.number_input("Learning Rate", 0.00001, 0.01, float(c_train.get('learning_rate', 0.0003)), format="%.5f")

        with col2:
            batch_size = st.number_input("Batch Size", 16, 256, int(c_train.get('batch_size', 64)))
            gamma = st.slider("Gamma (Discount Factor)", 0.9, 0.999, float(c_train.get('gamma', 0.99)))
            entropy_coef = st.number_input("Entropy Coefficient", 0.0, 0.1, float(c_train.get('ent_coef', 0.01)), format="%.3f")

        st.info(f"""
        **Algorithm Selected:** {algorithm}

        **{algorithm} Overview:**
        {"PPO: Proximal Policy Optimization - Stable on-policy algorithm" if algorithm == "PPO" else ""}
        {"DQN: Deep Q-Network - Off-policy value-based algorithm" if algorithm == "DQN" else ""}
        {"A3C: Asynchronous Advantage Actor-Critic - Multi-threaded policy gradient" if algorithm == "A3C" else ""}
        """)

    with tab3:
        st.subheader("Reward Function Design")

        st.write("""
        The reward function is crucial for learning optimal behavior.
        Adjust the weights to balance different objectives:
        """)

        col1, col2 = st.columns(2)

        with col1:
            delivery_reward = st.number_input("Delivery Success Reward", 0.0, 200.0, float(c_reward.get('delivery_success', 100.0)))
            efficiency_bonus = st.number_input("Efficiency Bonus", 0.0, 50.0, float(c_reward.get('efficiency_bonus', 10.0)))

        with col2:
            distance_penalty = st.number_input("Distance Penalty", -1.0, 0.0, float(c_reward.get('distance_penalty', -0.1)))
            collision_penalty = st.number_input("Collision Penalty", -100.0, 0.0, float(c_reward.get('collision_penalty', -50.0)))
            idle_penalty = st.number_input("Idle Penalty", -5.0, 0.0, float(c_reward.get('idle_penalty', -0.5)))

        st.write("**Reward Function Formula:**")
        st.latex(r"R = R_{delivery} + R_{efficiency} + R_{distance} + R_{collision} + R_{idle}")

        # Show example calculation
        st.write("**Example Calculation:**")
        total_reward = delivery_reward + efficiency_bonus + (distance_penalty * 10) + collision_penalty * 0 + idle_penalty * 5
        st.write(f"Total Reward (example): {total_reward:.2f}")

    # Initialization Button (Global for all tabs)
    st.markdown("---")
    if st.button("Initialize Environment", key="init_env"):
        # Create config
        config = {
            'warehouse': {
                'width': warehouse_width,
                'height': warehouse_height,
                'num_shelves': num_shelves,
                'num_pallets': num_pallets,
                'num_lgvs': num_lgvs
            },
            'lgv': {
                'max_speed': max_speed,
                'max_acceleration': max_accel,
                'turning_radius': turning_radius,
                'load_capacity': load_capacity,
                'loading_time': loading_time,
                'unloading_time': unloading_time
            },
            'rewards': {
                'delivery_success': delivery_reward,
                'distance_penalty': distance_penalty,
                'collision_penalty': collision_penalty,
                'idle_penalty': idle_penalty,
                'efficiency_bonus': efficiency_bonus
            },
            'training': {
                'algorithm': algorithm,
                'total_timesteps': total_timesteps,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'gamma': gamma,
                'n_steps': 2048,
                'ent_coef': entropy_coef
            }
        }

        # Update session state config
        st.session_state.config = config

        # Save config
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)

        # Initialize environment
        st.session_state.env = WarehouseEnv('config.yaml')
        st.success("✅ Environment initialized successfully!")

        # Show preview
        env_state = st.session_state.env.get_state()
        fig = Plotter.plot_warehouse_layout(env_state)
        st.plotly_chart(fig, use_container_width=True)


def show_training_page():
    """Training page"""
    st.header("🎓 Training Dashboard")

    if st.session_state.env is None:
        st.warning("⚠️ Please initialize the environment in Configuration page first!")
        return

    tab1, tab2, tab3 = st.tabs(["RL Training", "Baseline Algorithms", "Data Generation"])

    with tab1:
        st.subheader("Train RL Agents")

        col1, col2 = st.columns([2, 1])

        c_train = st.session_state.config.get('training', {})
        
        with col1:
            # Determine index for algorithm from config if possible
            algo_bg = c_train.get('algorithm', 'PPO')
            algo_opts = ["PPO", "DQN", "A3C"]
            algo_idx = algo_opts.index(algo_bg) if algo_bg in algo_opts else 0
            
            algorithm = st.selectbox("Select Algorithm", algo_opts, index=algo_idx, key="train_algo")
            timesteps = st.number_input("Training Timesteps", 10000, 1000000, int(c_train.get('total_timesteps', 100000)), 10000, key="train_steps")

        with col2:
            model_name = st.text_input("Model Name", f"{algorithm}_model")

        if st.button("Start Training", key="start_train"):
            with st.spinner(f"Training {algorithm} agent..."):
                # Initialize trainer
                trainer = Trainer('config.yaml')
                trainer.setup_agent(algorithm)

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Train
                results = trainer.train(save_path=f"models/{model_name}.pth", total_timesteps=timesteps)

                progress_bar.progress(100)
                status_text.text("Training complete!")

                # Store in session state
                st.session_state.trained_models[model_name] = {
                    'trainer': trainer,
                    'algorithm': algorithm,
                    'results': results
                }

                st.success(f"✅ {algorithm} training completed!")

                # Show training curves
                if results.get('rewards'):
                    fig = Plotter.plot_training_curves(
                        {'rewards': results['rewards'],
                         'episode_lengths': results.get('episode_lengths', [])},
                        algorithm
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Show metrics
                st.subheader("Training Metrics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Mean Reward", f"{results.get('mean_reward', 0):.2f}")
                with col2:
                    st.metric("Mean Length", f"{results.get('mean_length', 0):.0f}")
                with col3:
                    st.metric("Total Episodes", len(results.get('rewards', [])))
                with col4:
                    st.metric("Algorithm", algorithm)

        # Show trained models
        if st.session_state.trained_models:
            st.subheader("Trained Models")
            models_df = pd.DataFrame([
                {
                    'Name': name,
                    'Algorithm': info['algorithm'],
                    'Mean Reward': info['results'].get('mean_reward', 0)
                }
                for name, info in st.session_state.trained_models.items()
            ])
            st.dataframe(models_df, use_container_width=True)

    with tab2:
        st.subheader("Baseline Algorithms")

        st.write("""
        Compare RL agents with traditional pathfinding algorithms:
        - **A***: Optimal pathfinding with heuristic
        - **Dijkstra**: Guaranteed shortest path
        - **Greedy**: Fast but suboptimal
        - **Random**: Baseline comparison
        """)

        baseline_algo = st.selectbox("Select Baseline", ["A_star", "Dijkstra", "Greedy", "Random"])
        num_episodes = st.number_input("Number of Episodes", 10, 500, 100, key="baseline_episodes")

        if st.button("Run Baseline", key="run_baseline"):
            with st.spinner(f"Running {baseline_algo}..."):
                runner = BaselineRunner(st.session_state.env)
                results = runner.run_algorithm(baseline_algo, num_episodes)

                st.session_state.evaluation_results[baseline_algo] = results

                st.success(f"✅ {baseline_algo} completed!")

                # Show results
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Mean Reward", f"{results['mean_reward']:.2f}")
                with col2:
                    st.metric("Mean Length", f"{results['mean_episode_length']:.0f}")
                with col3:
                    st.metric("Completion Rate", f"{results['mean_completion_rate']:.1%}")
                with col4:
                    st.metric("Mean Distance", f"{results['mean_distance']:.2f}")

        # Run all baselines
        if st.button("Run All Baselines", key="run_all_baselines"):
            with st.spinner("Running all baseline algorithms..."):
                runner = BaselineRunner(st.session_state.env)

                for algo in ["A_star", "Dijkstra", "Greedy", "Random"]:
                    results = runner.run_algorithm(algo, num_episodes)
                    st.session_state.evaluation_results[algo] = results

                st.success("✅ All baselines completed!")

    with tab3:
        st.subheader("Data Generation")

        st.write("""
        Generate synthetic data for analysis and offline learning:
        """)

        data_type = st.selectbox("Data Type", ["Random Episodes", "Expert Demonstrations", "State-Action Pairs"])
        num_samples = st.number_input("Number of Samples", 100, 10000, 1000)

        if st.button("Generate Data", key="gen_data"):
            with st.spinner("Generating data..."):
                generator = DataGenerator(st.session_state.env)

                if data_type == "Random Episodes":
                    data = generator.generate_random_episodes(num_samples)
                    st.success(f"✅ Generated {len(data)} episodes")
                    st.dataframe(data.head(20))

                    # Save option
                    if st.button("Save Dataset"):
                        generator.save_dataset(data, f"data/random_episodes_{num_samples}.csv")
                        st.success("Dataset saved!")

                elif data_type == "Expert Demonstrations":
                    planner_type = st.selectbox("Planner Type", ["A_star", "Dijkstra"])
                    data = generator.generate_expert_demonstrations(planner_type, num_samples)
                    st.success(f"✅ Generated expert demonstrations")
                    st.json(data)


def show_evaluation_page():
    """Evaluation page"""
    st.header("📊 Evaluation & Comparison")

    if not st.session_state.trained_models and not st.session_state.evaluation_results:
        st.warning("⚠️ No trained models or evaluation results available. Please train models first!")
        return

    tab1, tab2, tab3 = st.tabs(["Model Evaluation", "Algorithm Comparison", "Statistical Analysis"])

    with tab1:
        st.subheader("Evaluate Trained Models")

        if st.session_state.trained_models:
            model_name = st.selectbox("Select Model", list(st.session_state.trained_models.keys()))
            num_eval_episodes = st.number_input("Evaluation Episodes", 10, 500, 100, key="eval_episodes")

            if st.button("Evaluate Model"):
                with st.spinner("Evaluating..."):
                    model_info = st.session_state.trained_models[model_name]
                    trainer = model_info['trainer']

                    evaluator = Evaluator(st.session_state.env)
                    results = evaluator.evaluate_rl_agent(trainer, num_eval_episodes)

                    st.session_state.evaluation_results[model_name] = results

                    st.success("✅ Evaluation complete!")

                    # Display metrics
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Mean Reward", f"{results['mean_reward']:.2f}")
                    with col2:
                        st.metric("Completion Rate", f"{results.get('mean_completion_rate', 0):.1%}")
                    with col3:
                        st.metric("Mean Distance", f"{results.get('mean_distance', 0):.2f}")
                    with col4:
                        st.metric("Collisions", f"{results.get('mean_collisions', 0):.2f}")
                    with col5:
                        st.metric("Efficiency Score", f"{results.get('efficiency_score', 0):.1f}/100")

                    # Show episode rewards distribution
                    if 'episode_rewards' in results:
                        st.subheader("Reward Distribution")
                        fig = Plotter.plot_metrics_over_time(results)
                        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Algorithm Comparison")

        if len(st.session_state.evaluation_results) < 2:
            st.info("Train and evaluate at least 2 algorithms to compare.")
        else:
            # Create comparison
            analyzer = ComparisonAnalyzer()

            for name, results in st.session_state.evaluation_results.items():
                analyzer.add_results(name, results)

            comparison_df = analyzer.compare_algorithms()

            st.subheader("Performance Comparison Table")
            st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['efficiency_score']),
                        use_container_width=True)

            # Visualization
            st.subheader("Visual Comparison")
            fig = Plotter.plot_comparison(comparison_df)
            st.plotly_chart(fig, use_container_width=True)

            # Radar chart
            st.subheader("Multi-Dimensional Comparison")
            fig_radar = Plotter.plot_radar_comparison(st.session_state.evaluation_results)
            st.plotly_chart(fig_radar, use_container_width=True)

            # Best algorithm
            st.success(f"🏆 Best Algorithm: **{analyzer.best_algorithm()}**")

    with tab3:
        st.subheader("Statistical Significance Testing")

        if len(st.session_state.evaluation_results) >= 2:
            algos = list(st.session_state.evaluation_results.keys())

            col1, col2 = st.columns(2)

            with col1:
                algo1 = st.selectbox("Algorithm 1", algos, key="stat_algo1")
            with col2:
                algo2 = st.selectbox("Algorithm 2", [a for a in algos if a != algo1], key="stat_algo2")

            if st.button("Run Statistical Test"):
                analyzer = ComparisonAnalyzer()
                for name, results in st.session_state.evaluation_results.items():
                    analyzer.add_results(name, results)

                test_result = analyzer.statistical_significance_test(algo1, algo2)

                st.subheader("T-Test Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(f"{algo1} Mean", f"{test_result['mean_1']:.2f}")
                with col2:
                    st.metric(f"{algo2} Mean", f"{test_result['mean_2']:.2f}")
                with col3:
                    st.metric("P-Value", f"{test_result['p_value']:.4f}")

                if test_result['significant']:
                    st.success(f"✅ Statistically significant difference (p < 0.05)")
                else:
                    st.info(f"❌ No statistically significant difference (p ≥ 0.05)")

                st.write(f"**Effect Size (Cohen's d):** {test_result['cohens_d']:.3f} ({test_result['effect_size']})")


def show_simulation_page():
    """Simulation page"""
    st.header("🎮 Real-Time Simulation")

    if st.session_state.env is None:
        st.warning("⚠️ Please initialize the environment first!")
        return

    tab1, tab2 = st.tabs(["Live Simulation", "Replay & Analysis"])

    with tab1:
        st.subheader("Live Warehouse Simulation")

        # Control panel
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.trained_models:
                control_mode = st.selectbox("Control Mode", ["Trained Model", "Baseline", "Manual"])
            else:
                control_mode = st.selectbox("Control Mode", ["Baseline", "Manual"])

        with col2:
            if control_mode == "Trained Model":
                model_name = st.selectbox("Select Model", list(st.session_state.trained_models.keys()))
            elif control_mode == "Baseline":
                baseline = st.selectbox("Select Baseline", ["A_star", "Dijkstra", "Greedy"])

        with col3:
            num_steps = st.number_input("Simulation Steps", 10, 1000, 100)

        if st.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                # Reset environment
                observations, _ = st.session_state.env.reset()

                # Create animator
                animator = WarehouseAnimator(
                    st.session_state.env.width,
                    st.session_state.env.height
                )

                # Initialize trajectory tracking
                trajectories = {i: [] for i in range(st.session_state.env.num_lgvs)}

                # Run simulation
                for step in range(num_steps):
                    # Get actions based on control mode
                    if control_mode == "Trained Model":
                        model_info = st.session_state.trained_models[model_name]
                        trainer = model_info['trainer']
                        actions = {}
                        for agent_id in range(st.session_state.env.num_lgvs):
                            action, _ = trainer.agent.predict(observations[agent_id], deterministic=True)
                            actions[agent_id] = action

                    elif control_mode == "Baseline":
                        # Use baseline planner
                        from src.baselines.pathfinding import AStarPlanner, DijkstraPlanner, GreedyPlanner

                        planner_map = {
                            'A_star': AStarPlanner,
                            'Dijkstra': DijkstraPlanner,
                            'Greedy': GreedyPlanner
                        }

                        planner = planner_map[baseline](st.session_state.env.grid)
                        runner = BaselineRunner(st.session_state.env)
                        actions = runner._execute_plans(runner._plan_all_lgvs(planner))

                    else:  # Manual - random
                        actions = {i: st.session_state.env.action_space.sample()
                                 for i in range(st.session_state.env.num_lgvs)}

                    # Step environment
                    observations, rewards, dones, truncated, info = st.session_state.env.step(actions)

                    # Record state
                    state = st.session_state.env.get_state()
                    animator.add_frame(state)

                    # Track trajectories
                    for lgv in state['lgvs']:
                        trajectories[lgv['id']].append(lgv['position'])

                    if dones.get('__all__', False):
                        break

                st.success(f"✅ Simulation complete! {step + 1} steps")

                # Show final state
                st.subheader("Final Warehouse State")
                final_state = st.session_state.env.get_state()
                fig = Plotter.plot_warehouse_layout(final_state)
                st.plotly_chart(fig, use_container_width=True)

                # Show trajectories
                st.subheader("LGV Trajectories")
                fig_traj = WarehouseAnimator.create_trajectory_plot(trajectories)
                st.plotly_chart(fig_traj, use_container_width=True)

                # Show statistics
                st.subheader("Simulation Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Deliveries", info['total_deliveries'])
                with col2:
                    st.metric("Total Distance", f"{info['total_distance']:.2f}")
                with col3:
                    st.metric("Collisions", info['total_collisions'])
                with col4:
                    st.metric("Completion Rate", f"{info['completion_rate']:.1%}")

                # Show LGV stats
                st.subheader("Individual LGV Performance")
                fig_lgv = Plotter.plot_lgv_statistics(final_state)
                st.plotly_chart(fig_lgv, use_container_width=True)

                # Save simulation to history
                sim_record = {
                    'algorithm': control_mode if control_mode == "Manual" else (model_name if control_mode == "Trained Model" else baseline),
                    'steps': step + 1,
                    'final_reward': sum(rewards.values()) if rewards else 0,
                    'deliveries': info['total_deliveries'],
                    'trajectories': animator.frames[0]['lgvs'] if animator.frames else {},
                    'frames': animator.frames, # Save all frames for animated replay
                    'final_state': final_state,
                    'info': info
                }
                st.session_state.simulation_history.append(sim_record)
                st.success(f"Simulation saved to history! Total recorded: {len(st.session_state.simulation_history)}")

    with tab2:
        st.subheader("Simulation Replay & Analysis")

        # Show recorded simulations from session state
        if st.session_state.simulation_history:
            st.write(f"**Recorded Simulations:** {len(st.session_state.simulation_history)}")

            # Select simulation to replay
            sim_idx = st.selectbox(
                "Select Simulation",
                range(len(st.session_state.simulation_history)),
                format_func=lambda i: f"Simulation {i+1} - {st.session_state.simulation_history[i].get('algorithm', 'Unknown')} ({st.session_state.simulation_history[i].get('steps', 0)} steps)"
            )

            if st.button("Replay Selected Simulation"):
                sim_data = st.session_state.simulation_history[sim_idx]

                st.write("### Simulation Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Algorithm", sim_data.get('algorithm', 'N/A'))
                with col2:
                    st.metric("Total Steps", sim_data.get('steps', 0))
                with col3:
                    st.metric("Final Reward", f"{sim_data.get('final_reward', 0):.2f}")
                with col4:
                    st.metric("Deliveries", sim_data.get('deliveries', 0))

                # Show trajectory visualization
                # Show trajectory visualization or animation
                if 'frames' in sim_data and sim_data['frames']:
                    st.subheader("🔁 Animated Replay")
                    st.info("Click 'Play' to watch the LGV journey.")
                    
                    # Reconstruct animator to generate plot
                    replay_animator = WarehouseAnimator(st.session_state.env.width, st.session_state.env.height)
                    replay_animator.frames = sim_data['frames']
                    
                    fig_anim = replay_animator.create_plotly_animation()
                    st.plotly_chart(fig_anim, use_container_width=True)
                    
                elif 'trajectories' in sim_data:
                    st.subheader("LGV Trajectories (Static)")
                    fig_traj = WarehouseAnimator.create_trajectory_plot(sim_data['trajectories'])
                    st.plotly_chart(fig_traj, use_container_width=True)

                # Show final state
                if 'final_state' in sim_data:
                    st.subheader("Final Warehouse State")
                    fig_final = Plotter.plot_warehouse_layout(sim_data['final_state'])
                    st.plotly_chart(fig_final, use_container_width=True)

            # Clear history button
            if st.button("Clear All Recorded Simulations"):
                st.session_state.simulation_history = []
                st.success("Simulation history cleared!")
                st.rerun()

        else:
            st.info("No recorded simulations yet. Run simulations in the Live Simulation tab to record them.")


def show_analysis_page():
    """Analysis page"""
    st.header("📈 Advanced Analysis")

    if not st.session_state.evaluation_results:
        st.warning("⚠️ No evaluation results available. Please evaluate models first!")
        return

    tab1, tab2, tab3 = st.tabs(["Performance Analysis", "Learning Curves", "Insights & Recommendations"])

    with tab1:
        st.subheader("Detailed Performance Analysis")

        # Summary statistics
        st.write("### Summary Statistics")

        summary_data = []
        for name, results in st.session_state.evaluation_results.items():
            summary_data.append({
                'Algorithm': name,
                'Mean Reward': results.get('mean_reward', 0),
                'Std Reward': results.get('std_reward', 0),
                'Completion Rate': results.get('mean_completion_rate', 0),
                'Efficiency Score': results.get('efficiency_score', 0),
                'Mean Distance': results.get('mean_distance', 0),
                'Mean Collisions': results.get('mean_collisions', 0)
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # Detailed metrics
        st.write("### Metric Breakdown")

        selected_algo = st.selectbox("Select Algorithm", list(st.session_state.evaluation_results.keys()))
        results = st.session_state.evaluation_results[selected_algo]

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Reward Metrics**")
            st.write(f"Mean: {results.get('mean_reward', 0):.2f}")
            st.write(f"Std: {results.get('std_reward', 0):.2f}")
            st.write(f"Min: {results.get('min_reward', 0):.2f}")
            st.write(f"Max: {results.get('max_reward', 0):.2f}")

        with col2:
            st.write("**Efficiency Metrics**")
            st.write(f"Completion Rate: {results.get('mean_completion_rate', 0):.1%}")
            st.write(f"Efficiency Score: {results.get('efficiency_score', 0):.1f}/100")
            st.write(f"Mean Distance: {results.get('mean_distance', 0):.2f}")

    with tab2:
        st.subheader("Learning Curve Analysis")

        if st.session_state.trained_models:
            model_name = st.selectbox("Select Model", list(st.session_state.trained_models.keys()), key="lc_model")

            model_info = st.session_state.trained_models[model_name]
            trainer = model_info['trainer']

            evaluator = Evaluator(st.session_state.env)
            analysis = evaluator.get_learning_curve_analysis(trainer)

            if 'error' not in analysis:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Initial Performance", f"{analysis['initial_performance']:.2f}")
                with col2:
                    st.metric("Final Performance", f"{analysis['final_performance']:.2f}")
                with col3:
                    st.metric("Improvement", f"{analysis['improvement']:.1f}%")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Convergence Episode", analysis['convergence_episode'])
                with col2:
                    st.metric("Stability (Std)", f"{analysis['stability']:.2f}")

                # Plot learning curve
                training_data = trainer.get_training_curves()
                if training_data:
                    fig = Plotter.plot_training_curves(training_data, model_name)
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Insights & Recommendations")

        st.write("### Key Findings")

        # Generate insights
        if st.session_state.evaluation_results:
            analyzer = ComparisonAnalyzer()
            for name, results in st.session_state.evaluation_results.items():
                analyzer.add_results(name, results)

            best_algo = analyzer.best_algorithm()

            st.success(f"🏆 **Best Performing Algorithm:** {best_algo}")

            best_results = st.session_state.evaluation_results[best_algo]

            st.write(f"""
            **Performance Highlights:**
            - Mean Reward: {best_results.get('mean_reward', 0):.2f}
            - Completion Rate: {best_results.get('mean_completion_rate', 0):.1%}
            - Efficiency Score: {best_results.get('efficiency_score', 0):.1f}/100
            """)

            st.write("### Recommendations")

            # Generate recommendations based on results
            recommendations = []

            # Check completion rate
            completion_rate = best_results.get('mean_completion_rate', 0)
            if completion_rate < 0.5:
                recommendations.append("⚠️ Low completion rate. Consider adjusting reward function or increasing training time.")
            elif completion_rate > 0.8:
                recommendations.append("✅ High completion rate achieved!")

            # Check collisions
            collisions = best_results.get('mean_collisions', 0)
            if collisions > 5:
                recommendations.append("⚠️ High collision rate. Consider increasing collision penalty or improving obstacle avoidance.")
            elif collisions < 2:
                recommendations.append("✅ Low collision rate - good safety performance!")

            # Check efficiency
            efficiency = best_results.get('efficiency_score', 0)
            if efficiency < 50:
                recommendations.append("⚠️ Low efficiency score. Review reward function and training hyperparameters.")
            elif efficiency > 70:
                recommendations.append("✅ High efficiency achieved!")

            for rec in recommendations:
                st.write(rec)

            st.write("### Future Improvements")
            st.write("""
            - Fine-tune hyperparameters for better performance
            - Experiment with different reward function weights
            - Increase training duration for better convergence
            - Try ensemble methods combining multiple algorithms
            - Implement curriculum learning for gradual difficulty increase
            """)


def show_research_page():
    """Research methodology page"""
    st.header("📚 Research Methodology & Documentation")

    tab1, tab2, tab3, tab4 = st.tabs(["Methodology", "Algorithms", "Metrics", "References"])

    with tab1:
        st.subheader("Research Methodology")

        st.write("""
        ### 1. Problem Definition

        **Objective:** Optimize the movement and task sequencing of automated guided vehicles (LGVs)
        in a warehouse environment using multi-agent reinforcement learning.

        **Key Challenges:**
        - Multi-agent coordination
        - Dynamic environment
        - Kinematic constraints
        - Collision avoidance
        - Task prioritization

        ### 2. Experimental Design

        **Environment Setup:**
        - Warehouse dimensions: Configurable grid
        - Number of LGVs: 2-10
        - Number of pallets: 10-100
        - Physical constraints: Speed, acceleration, turning radius

        **Training Methodology:**
        1. Initialize environment with random configurations
        2. Train multiple RL algorithms (PPO, DQN, A3C)
        3. Compare with baseline algorithms (A*, Dijkstra)
        4. Evaluate on standardized test scenarios
        5. Statistical significance testing

        ### 3. Reward Function Design

        The reward function balances multiple objectives:
        """)

        st.latex(r"R_t = w_1 R_{delivery} + w_2 R_{efficiency} + w_3 R_{distance} + w_4 R_{collision} + w_5 R_{idle}")

        st.write("""
        Where:
        - $R_{delivery}$: Reward for successful delivery
        - $R_{efficiency}$: Bonus for efficient movement
        - $R_{distance}$: Penalty for distance traveled
        - $R_{collision}$: Penalty for collisions
        - $R_{idle}$: Penalty for idle time

        ### 4. Evaluation Metrics

        - **Task Completion Rate**: Percentage of successful deliveries
        - **Total Distance**: Cumulative distance traveled
        - **Collision Rate**: Collisions per episode
        - **Average Delivery Time**: Time from pickup to delivery
        - **Efficiency Score**: Composite metric (0-100)
        """)

    with tab2:
        st.subheader("RL Algorithms")

        algo_selection = st.selectbox("Select Algorithm", ["PPO", "DQN", "A3C", "A*", "Dijkstra"])

        if algo_selection == "PPO":
            st.write("""
            ### Proximal Policy Optimization (PPO)

            **Type:** On-policy actor-critic algorithm

            **Key Features:**
            - Clipped surrogate objective for stable training
            - Multiple epochs of minibatch updates
            - Advantage estimation using GAE

            **Objective Function:**
            """)
            st.latex(r"L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]")

            st.write("""
            **Advantages:**
            - Stable training
            - Good sample efficiency
            - Works well in continuous and discrete spaces

            **Hyperparameters:**
            - Learning rate: 3e-4
            - Clip range: 0.2
            - GAE lambda: 0.95
            - Entropy coefficient: 0.01
            """)

        elif algo_selection == "DQN":
            st.write("""
            ### Deep Q-Network (DQN)

            **Type:** Off-policy value-based algorithm

            **Key Features:**
            - Experience replay buffer
            - Target network for stability
            - Epsilon-greedy exploration

            **Q-Learning Update:**
            """)
            st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'}Q(s',a') - Q(s,a)]")

            st.write("""
            **Advantages:**
            - Sample efficient (off-policy)
            - Stable with experience replay
            - Good for discrete action spaces

            **Hyperparameters:**
            - Learning rate: 1e-4
            - Epsilon: 1.0 → 0.01 (decay)
            - Replay buffer size: 100,000
            - Target update frequency: 1000 steps
            """)

        elif algo_selection == "A3C":
            st.write("""
            ### Asynchronous Advantage Actor-Critic (A3C)

            **Type:** On-policy actor-critic algorithm

            **Key Features:**
            - Asynchronous parallel training
            - N-step returns
            - Entropy regularization

            **Advantage Function:**
            """)
            st.latex(r"A(s,a) = Q(s,a) - V(s) = \sum_{i=0}^{n-1}\gamma^i r_{t+i} + \gamma^n V(s_{t+n}) - V(s_t)")

            st.write("""
            **Advantages:**
            - Fast training with parallelization
            - Good exploration
            - Stable learning

            **Hyperparameters:**
            - Learning rate: 1e-4
            - N-steps: 20
            - Entropy coefficient: 0.01
            """)

        elif algo_selection == "A*":
            st.write("""
            ### A* Pathfinding

            **Type:** Informed search algorithm

            **Algorithm:**
            """)
            st.latex(r"f(n) = g(n) + h(n)")

            st.write("""
            Where:
            - $g(n)$: Cost from start to node n
            - $h(n)$: Heuristic estimate from n to goal

            **Heuristic:** Manhattan distance

            **Properties:**
            - Optimal with admissible heuristic
            - Complete
            - Time complexity: O(b^d)
            """)

        else:  # Dijkstra
            st.write("""
            ### Dijkstra's Algorithm

            **Type:** Uninformed search algorithm

            **Properties:**
            - Guaranteed shortest path
            - No heuristic needed
            - Time complexity: O((V + E) log V)

            **Advantages:**
            - Optimal
            - Complete
            - Deterministic

            **Disadvantages:**
            - Slower than A*
            - Explores more nodes
            """)

    with tab3:
        st.subheader("Evaluation Metrics")

        st.write("""
        ### Primary Metrics

        **1. Task Completion Rate**
        """)
        st.latex(r"\text{Completion Rate} = \frac{\text{Delivered Pallets}}{\text{Total Pallets}}")

        st.write("""
        **2. Average Delivery Time**
        """)
        st.latex(r"\text{Avg Delivery Time} = \frac{1}{N}\sum_{i=1}^{N}(t_{delivery,i} - t_{pickup,i})")

        st.write("""
        **3. Collision Rate**
        """)
        st.latex(r"\text{Collision Rate} = \frac{\text{Collisions}}{\text{Total Steps}}")

        st.write("""
        **4. Distance Efficiency**
        """)
        st.latex(r"\text{Distance Efficiency} = \frac{\text{Deliveries}}{\text{Total Distance}}")

        st.write("""
        **5. Efficiency Score (Composite)**
        """)
        st.latex(r"\text{Score} = 0.4 \times CR + 0.2 \times (1-CoR) + 0.2 \times DE + 0.2 \times TE")

        st.write("""
        Where:
        - CR: Completion Rate
        - CoR: Collision Rate
        - DE: Distance Efficiency
        - TE: Time Efficiency

        ### Statistical Analysis

        **T-Test for Significance:**
        """)
        st.latex(r"t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}")

        st.write("""
        **Effect Size (Cohen's d):**
        """)
        st.latex(r"d = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2 + s_2^2}{2}}}")

    with tab4:
        st.subheader("References & Resources")

        st.write("""
        ### Key Papers

        1. **Schulman, J., et al. (2017)**
           "Proximal Policy Optimization Algorithms"
           *arXiv preprint arXiv:1707.06347*

        2. **Mnih, V., et al. (2015)**
           "Human-level control through deep reinforcement learning"
           *Nature, 518(7540), 529-533*

        3. **Mnih, V., et al. (2016)**
           "Asynchronous methods for deep reinforcement learning"
           *ICML 2016*

        4. **Lowe, R., et al. (2017)**
           "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
           *NIPS 2017*

        ### Frameworks & Libraries

        - **PyTorch**: Deep learning framework
        - **Gymnasium**: RL environment interface
        - **Streamlit**: Dashboard framework
        - **Plotly**: Interactive visualizations

        ### Online Resources

        - [OpenAI Spinning Up](https://spinningup.openai.com/)
        - [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
        - [Gymnasium Documentation](https://gymnasium.farama.org/)

        ### Course Information

        **University of Naples**
        Reinforcement Learning Project 2026

        **Project Repository:**
        Located at: `/Users/shayan/Unina/rl_projs/marl_lgv_4/`
        """)


if __name__ == "__main__":
    main()
