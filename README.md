# Multi-Agent Reinforcement Learning for Warehouse LGV Optimization

<div align="center">

**🤖 Complete End-to-End MARL System with Streamlit Dashboard 🤖**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.26%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

</div>

---

## 📋 Project Overview

This project implements a **complete, professional, end-to-end** Multi-Agent Reinforcement Learning (MARL) system for optimizing automated guided vehicle (LGV) operations in warehouse environments.

### 🎯 Objectives

- **Path Planning**: Optimize LGV routing with kinematic constraints
- **Task Sequencing**: Efficient load pickup and delivery scheduling
- **Multi-Agent Coordination**: Collision avoidance and cooperation
- **Performance Analysis**: Comprehensive evaluation and comparison
- **Visualization**: Real-time simulation and trajectory analysis

### ✨ Key Features

- ✅ **Multi-Agent Warehouse Simulation** with dynamic pallet positioning
- ✅ **3 RL Algorithms**: PPO, DQN, A3C (fully implemented)
- ✅ **4 Baseline Algorithms**: A*, Dijkstra, Greedy, Random
- ✅ **Complete Training Pipeline** with model management
- ✅ **Comprehensive Evaluation** with statistical analysis
- ✅ **Interactive Streamlit Dashboard** (7 pages)
- ✅ **Real-Time Visualization** and animation
- ✅ **No Placeholders** - Everything fully functional!

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to project directory
cd /Users/shayan/Unina/rl_projs/marl_lgv_4

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run system test
python test_system.py
```

### Launch Streamlit Dashboard

```bash
# Method 1: Direct launch
streamlit run app.py

# Method 2: Quick start script
./run_app.sh
```

Then open your browser at: **http://localhost:8501**

### Command-Line Usage

```bash
# Train an RL agent
python run_training.py --algorithm PPO --timesteps 100000

# Evaluate trained model
python run_evaluation.py --model models/PPO_model.pth --episodes 100

# Evaluate baseline
python run_evaluation.py --baseline A_star --episodes 100
```

---

## 📁 Project Structure

```
marl_lgv_4/
├── app.py                      # Main Streamlit dashboard (1,089 lines)
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── QUICKSTART.md              # Quick start guide
├── PROJECT_SUMMARY.md         # Detailed project summary
├── test_system.py             # System integration test
├── run_training.py            # CLI training script
├── run_evaluation.py          # CLI evaluation script
├── run_app.sh                 # Quick start script
│
├── src/                       # Source code (5,000+ lines)
│   ├── environment/           # Warehouse simulation
│   │   ├── warehouse_env.py   # Multi-agent environment (471 lines)
│   │   └── entities.py        # LGV, Pallet, Shelf classes (123 lines)
│   │
│   ├── agents/                # RL algorithms
│   │   ├── ppo_agent.py       # PPO implementation (318 lines)
│   │   ├── dqn_agent.py       # DQN implementation (247 lines)
│   │   ├── a3c_agent.py       # A3C implementation (252 lines)
│   │   └── base_agent.py      # Base agent class (25 lines)
│   │
│   ├── baselines/             # Traditional pathfinding
│   │   ├── pathfinding.py     # A*, Dijkstra, Greedy, Random (154 lines)
│   │   └── baseline_runner.py # Baseline execution (207 lines)
│   │
│   ├── training/              # Training pipeline
│   │   ├── trainer.py         # Unified trainer (189 lines)
│   │   └── data_generator.py  # Data generation (190 lines)
│   │
│   ├── evaluation/            # Evaluation system
│   │   ├── evaluator.py       # Evaluator (235 lines)
│   │   └── metrics.py         # Metrics calculator (236 lines)
│   │
│   └── visualization/         # Plotting utilities
│       ├── plotter.py         # Interactive plots (346 lines)
│       └── animator.py        # Animation system (232 lines)
│
├── models/                    # Saved trained models
├── data/                      # Training data
└── results/                   # Experiment results
```

**Statistics:**
- 📊 **Total Files**: 57
- 🐍 **Python Files**: 25
- 📝 **Lines of Code**: 5,000+
- 🎯 **Algorithms**: 7 (3 RL + 4 baselines)
- 📱 **Dashboard Pages**: 7

---

## 🎮 Streamlit Dashboard Features

### 1. 🏠 Home Page
- Project overview and objectives
- System status monitoring
- Quick start guide
- Feature summary

### 2. ⚙️ Configuration Page
- **Warehouse Setup**: Size, shelves, pallets, LGVs
- **LGV Constraints**: Speed, acceleration, turning radius
- **RL Configuration**: Algorithm selection, hyperparameters
- **Reward Design**: Customizable reward function weights

### 3. 🎓 Training Page
- **RL Training**: Train PPO, DQN, or A3C agents
- **Baseline Algorithms**: Run A*, Dijkstra, Greedy, Random
- **Data Generation**: Create synthetic training datasets
- **Progress Monitoring**: Real-time training metrics
- **Model Management**: Save and load trained models

### 4. 📊 Evaluation Page
- **Model Evaluation**: Test trained agents
- **Algorithm Comparison**: Side-by-side performance comparison
- **Statistical Analysis**: T-tests, effect sizes, significance
- **Visual Comparison**: Interactive charts and radar plots
- **Efficiency Scoring**: Composite performance metrics

### 5. 🎮 Simulation Page
- **Live Simulation**: Real-time warehouse operations
- **Control Modes**: Trained Model, Baseline, or Manual
- **Visualization**: Interactive warehouse layout
- **Trajectory Tracking**: LGV path visualization
- **Statistics**: Individual LGV performance metrics

### 6. 📈 Analysis Page
- **Performance Analysis**: Detailed metric breakdown
- **Learning Curves**: Training progress visualization
- **Convergence Analysis**: Identify optimal training duration
- **Insights**: Automated recommendations
- **Export Results**: Save analysis data

### 7. 📚 Research Page
- **Methodology**: Complete research approach
- **Algorithm Details**: Mathematical formulations
- **Metrics Explanation**: Evaluation criteria
- **References**: Key papers and resources

---

## 🧠 Implemented Algorithms

### Reinforcement Learning

#### 1. **PPO (Proximal Policy Optimization)**
- Type: On-policy actor-critic
- Features: Clipped objective, GAE, multi-head actions
- Best for: Stable training, high performance
- Implementation: `src/agents/ppo_agent.py`

#### 2. **DQN (Deep Q-Network)**
- Type: Off-policy value-based
- Features: Experience replay, target network, ε-greedy
- Best for: Sample efficiency, discrete actions
- Implementation: `src/agents/dqn_agent.py`

#### 3. **A3C (Asynchronous Advantage Actor-Critic)**
- Type: On-policy actor-critic
- Features: N-step returns, entropy regularization
- Best for: Fast training with parallelization
- Implementation: `src/agents/a3c_agent.py`

### Baseline Algorithms

#### 4. **A* Pathfinding**
- Optimal pathfinding with Manhattan heuristic
- Guaranteed shortest path
- Fast computation with priority queue

#### 5. **Dijkstra's Algorithm**
- Guaranteed shortest path without heuristic
- Complete and optimal
- Uninformed search

#### 6. **Greedy Best-First**
- Heuristic-based fast search
- Suboptimal but quick
- Good for comparison

#### 7. **Random Planner**
- Random walk with goal bias
- Baseline for minimum performance
- Shows improvement potential

---

## 📊 Evaluation Metrics

### Primary Metrics
- **Task Completion Rate**: % of successful deliveries
- **Mean Reward**: Average cumulative reward per episode
- **Episode Length**: Average steps to completion
- **Total Distance**: Cumulative distance traveled by all LGVs

### Secondary Metrics
- **Collision Count**: Number of collisions per episode
- **Collision Rate**: Collisions per step
- **Average Delivery Time**: Time from pickup to delivery
- **Distance Efficiency**: Deliveries per distance unit
- **Idle Time**: Time spent waiting

### Composite Metrics
- **Efficiency Score** (0-100): Weighted combination of:
  - 40% Completion Rate
  - 20% Collision Avoidance
  - 20% Distance Efficiency
  - 20% Time Efficiency

### Statistical Analysis
- **T-Tests**: Statistical significance between algorithms
- **Effect Size (Cohen's d)**: Magnitude of differences
- **Confidence Intervals**: Result reliability
- **Distribution Analysis**: Performance variability

---

## 🔬 Research Methodology

### 1. Problem Definition
Multi-agent path planning and task sequencing for LGVs with:
- Dynamic environment
- Kinematic constraints
- Collision avoidance
- Task prioritization

### 2. Environment Design
- **State Space**: Position, velocity, load status, obstacles, other agents
- **Action Space**: Multi-discrete [5,5,2,2] for acceleration, steering, load, wait
- **Reward Function**: Balanced delivery, efficiency, safety objectives

### 3. Training Methodology
- Episode-based learning
- Experience replay (DQN)
- On-policy updates (PPO, A3C)
- Adaptive exploration

### 4. Evaluation Approach
- Multiple independent runs
- Statistical significance testing
- Baseline comparison
- Ablation studies

### 5. Analysis & Insights
- Learning curve analysis
- Convergence detection
- Performance breakdown
- Automated recommendations

---

## 📈 Example Results

### Performance Comparison (Example)

| Algorithm | Mean Reward | Completion Rate | Collisions | Efficiency Score |
|-----------|-------------|-----------------|------------|------------------|
| PPO       | 245.3 ± 12.1| 87.5%          | 1.2        | 78.4/100        |
| DQN       | 198.7 ± 15.3| 79.2%          | 2.3        | 68.9/100        |
| A3C       | 221.4 ± 14.0| 82.8%          | 1.8        | 72.1/100        |
| A*        | 156.2 ± 8.5 | 92.1%          | 0.8        | 65.3/100        |
| Dijkstra  | 142.8 ± 7.2 | 90.5%          | 1.1        | 62.7/100        |

*Note: These are example values. Actual results depend on configuration.*

---

## 🛠️ Configuration

Edit `config.yaml` to customize:

```yaml
warehouse:
  width: 20              # Warehouse width
  height: 20             # Warehouse height
  num_shelves: 15        # Number of shelves
  num_pallets: 30        # Number of pallets
  num_lgvs: 6           # Number of LGVs

lgv:
  max_speed: 2.0        # Maximum speed (m/s)
  max_acceleration: 0.5  # Max acceleration (m/s²)
  turning_radius: 1.0    # Turning radius (m)
  loading_time: 5        # Loading time (s)
  unloading_time: 5      # Unloading time (s)

training:
  algorithm: "PPO"       # PPO, DQN, or A3C
  total_timesteps: 100000
  learning_rate: 0.0003
  batch_size: 64
  gamma: 0.99

rewards:
  delivery_success: 100.0
  distance_penalty: -0.1
  collision_penalty: -50.0
  idle_penalty: -0.5
  efficiency_bonus: 10.0
```

---

## 📚 Documentation

- **README.md**: This file - project overview
- **QUICKSTART.md**: Detailed quick start guide
- **PROJECT_SUMMARY.md**: Complete feature summary
- **Code Documentation**: Inline comments and docstrings

---

## 🧪 Testing

Run the comprehensive system test:

```bash
python test_system.py
```

Tests include:
- ✅ Environment initialization
- ✅ Environment step execution
- ✅ RL agent prediction
- ✅ Baseline pathfinding
- ✅ Training pipeline
- ✅ Evaluation system
- ✅ Visualization generation

**All tests pass successfully!**

---

## 💻 System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- CPU sufficient (GPU optional for faster training)
- Modern web browser for Streamlit dashboard

### Dependencies
- PyTorch 2.0+
- Gymnasium 0.29+
- Streamlit 1.26+
- NumPy, Pandas, Matplotlib
- Plotly for interactive visualizations
- See `requirements.txt` for complete list

---

## 🎓 Educational Content

This project demonstrates:
- ✅ Complete MARL system design
- ✅ Multiple RL algorithms implementation
- ✅ Evaluation methodology
- ✅ Statistical analysis
- ✅ Professional visualization
- ✅ Software engineering best practices
- ✅ Research documentation

---

## 🚧 Usage Examples

### Example 1: Quick Training
```python
from src.training.trainer import Trainer

trainer = Trainer('config.yaml')
trainer.setup_agent('PPO')
results = trainer.train(save_path='models/ppo_model.pth')
print(f"Mean Reward: {results['mean_reward']:.2f}")
```

### Example 2: Evaluation
```python
from src.evaluation.evaluator import Evaluator
from src.environment.warehouse_env import WarehouseEnv

env = WarehouseEnv('config.yaml')
evaluator = Evaluator(env)
results = evaluator.evaluate_baseline('A_star', num_episodes=100)
print(f"Completion Rate: {results['mean_completion_rate']:.1%}")
```

### Example 3: Comparison
```python
from src.evaluation.metrics import ComparisonAnalyzer

analyzer = ComparisonAnalyzer()
analyzer.add_results('PPO', ppo_results)
analyzer.add_results('A_star', astar_results)
comparison = analyzer.compare_algorithms()
print(comparison)
```

---

## 🎯 Expected Outcomes (ALL ACHIEVED ✅)

- ✅ **Trained RL Models**: All 3 algorithms fully implemented
- ✅ **Path Optimization**: Efficient routing with constraints
- ✅ **Visualization**: Complete warehouse animation system
- ✅ **Performance Comparison**: Statistical analysis with baselines
- ✅ **Optimal Sequencing**: Task prioritization implemented
- ✅ **Dashboard**: All features in Streamlit app

---

## 🤝 Contributing

This is an academic project for University of Naples. For questions or collaboration:
- Review the Research page in the dashboard
- Check documentation files
- Examine code with inline comments

---

## 📄 License

Academic Project - University of Naples
Reinforcement Learning Course 2026

---

## 🙏 Acknowledgments

- **OpenAI** for Spinning Up RL resources
- **Stable Baselines3** for RL implementations reference
- **Gymnasium** for environment interface
- **Streamlit** for amazing dashboard framework

---

## 📞 Support

For issues or questions:
1. Check QUICKSTART.md for common solutions
2. Review documentation in dashboard Research page
3. Examine test_system.py for troubleshooting

---

## 🎉 Status

**✅ PROJECT COMPLETE**

All features implemented, tested, and documented.
No placeholders, no missing components.
Ready for immediate use!

---

<div align="center">

**Built with ❤️ for Warehouse Optimization Research**

University of Naples | 2026

[View Project](.) | [Quick Start](QUICKSTART.md) | [Summary](PROJECT_SUMMARY.md)

</div>
