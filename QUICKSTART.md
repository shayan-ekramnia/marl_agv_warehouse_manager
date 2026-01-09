# MARL Warehouse LGV Optimization - Quick Start Guide

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to project directory:**
```bash
cd /Users/shayan/Unina/rl_projs/marl_lgv_4
```

2. **Create virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run system test:**
```bash
python test_system.py
```

## Usage

### Option 1: Streamlit Dashboard (Recommended)

Launch the interactive web dashboard:

```bash
streamlit run app.py
```

Or use the quick start script:

```bash
./run_app.sh
```

Then open your browser at: http://localhost:8501

### Option 2: Command Line Interface

**Train an RL agent:**
```bash
python run_training.py --algorithm PPO --timesteps 100000
```

**Evaluate a trained model:**
```bash
python run_evaluation.py --model models/PPO_model.pth --episodes 100
```

**Evaluate a baseline algorithm:**
```bash
python run_evaluation.py --baseline A_star --episodes 100
```

## Streamlit Dashboard Guide

### 1. Home Page
- Project overview
- System status
- Quick start guide

### 2. Configuration
- Set warehouse parameters (size, shelves, pallets, LGVs)
- Configure RL hyperparameters
- Design reward function
- Initialize environment

### 3. Training
- **RL Training**: Train PPO, DQN, or A3C agents
- **Baseline Algorithms**: Run A*, Dijkstra, Greedy, Random
- **Data Generation**: Create synthetic datasets

### 4. Evaluation
- Evaluate trained models
- Compare multiple algorithms
- Statistical significance testing
- Performance metrics

### 5. Simulation
- Run real-time simulations
- Visualize LGV movements
- Track trajectories
- View statistics

### 6. Analysis
- Detailed performance analysis
- Learning curve visualization
- Insights and recommendations
- Export results

### 7. Research
- Research methodology
- Algorithm documentation
- Evaluation metrics
- References

## Project Structure

```
marl_lgv_4/
├── app.py                  # Main Streamlit dashboard
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── QUICKSTART.md         # This file
├── test_system.py        # System integration test
├── run_training.py       # CLI training script
├── run_evaluation.py     # CLI evaluation script
├── run_app.sh           # Quick start script
│
├── src/                  # Source code
│   ├── environment/      # Warehouse simulation
│   │   ├── warehouse_env.py
│   │   └── entities.py
│   ├── agents/          # RL algorithms
│   │   ├── ppo_agent.py
│   │   ├── dqn_agent.py
│   │   └── a3c_agent.py
│   ├── baselines/       # Traditional algorithms
│   │   ├── pathfinding.py
│   │   └── baseline_runner.py
│   ├── training/        # Training pipeline
│   │   ├── trainer.py
│   │   └── data_generator.py
│   ├── evaluation/      # Evaluation system
│   │   ├── evaluator.py
│   │   └── metrics.py
│   └── visualization/   # Plotting utilities
│       ├── plotter.py
│       └── animator.py
│
├── models/              # Saved models
├── data/               # Training data
└── results/            # Experiment results
```

## Example Workflow

### 1. Setup and Configuration

```bash
# Run system test
python test_system.py

# Launch dashboard
streamlit run app.py
```

### 2. Configure Environment
- Go to Configuration page
- Set warehouse parameters:
  - Width: 20, Height: 20
  - Shelves: 15, Pallets: 30
  - LGVs: 6
- Click "Initialize Environment"

### 3. Train Models
- Go to Training page
- Select algorithm (e.g., PPO)
- Set timesteps: 100,000
- Click "Start Training"
- Wait for training to complete

### 4. Run Baselines
- In Training page, go to "Baseline Algorithms" tab
- Click "Run All Baselines"
- Wait for completion

### 5. Evaluate and Compare
- Go to Evaluation page
- Select trained model
- Click "Evaluate Model"
- View comparison with baselines

### 6. Run Simulation
- Go to Simulation page
- Select control mode (Trained Model or Baseline)
- Set simulation steps
- Click "Run Simulation"
- View warehouse visualization and trajectories

### 7. Analyze Results
- Go to Analysis page
- View detailed performance metrics
- Check learning curves
- Read insights and recommendations

## Configuration Options

### Warehouse Parameters
- `width`, `height`: Warehouse dimensions (10-50)
- `num_shelves`: Number of storage shelves (5-30)
- `num_pallets`: Number of pallets to transport (10-100)
- `num_lgvs`: Number of automated vehicles (2-10)

### LGV Physical Constraints
- `max_speed`: Maximum speed in m/s (0.5-5.0)
- `max_acceleration`: Maximum acceleration (0.1-2.0)
- `turning_radius`: Minimum turning radius (0.5-3.0)
- `loading_time`: Time to load pallet in seconds (1-20)
- `unloading_time`: Time to unload pallet (1-20)

### RL Training Parameters
- `algorithm`: PPO, DQN, or A3C
- `total_timesteps`: Training duration (10k-1M)
- `learning_rate`: Learning rate (1e-5 to 1e-2)
- `batch_size`: Mini-batch size (16-256)
- `gamma`: Discount factor (0.9-0.999)
- `ent_coef`: Entropy coefficient (0-0.1)

### Reward Function Weights
- `delivery_success`: Reward for successful delivery (0-200)
- `efficiency_bonus`: Bonus for efficient behavior (0-50)
- `distance_penalty`: Penalty per distance unit (-1 to 0)
- `collision_penalty`: Penalty for collisions (-100 to 0)
- `idle_penalty`: Penalty for idle time (-5 to 0)

## Algorithms

### RL Algorithms

**PPO (Proximal Policy Optimization)**
- Type: On-policy actor-critic
- Best for: Stable training, continuous improvement
- Training time: Medium
- Performance: High

**DQN (Deep Q-Network)**
- Type: Off-policy value-based
- Best for: Sample efficiency, discrete actions
- Training time: Fast
- Performance: Good

**A3C (Asynchronous Advantage Actor-Critic)**
- Type: On-policy actor-critic
- Best for: Fast parallel training
- Training time: Fast (with parallelization)
- Performance: Good

### Baseline Algorithms

**A***
- Optimal pathfinding with heuristic
- Guaranteed shortest path
- Fast computation

**Dijkstra**
- Guaranteed shortest path
- No heuristic needed
- Slower than A*

**Greedy**
- Fast but suboptimal
- Good for comparison
- Simple implementation

**Random**
- Baseline comparison
- Shows minimum performance
- Quick to run

## Performance Metrics

- **Mean Reward**: Average cumulative reward per episode
- **Completion Rate**: Percentage of successful deliveries
- **Episode Length**: Average steps per episode
- **Total Distance**: Cumulative distance traveled
- **Collision Count**: Number of collisions
- **Efficiency Score**: Composite metric (0-100)
- **Delivery Time**: Average time from pickup to delivery

## Troubleshooting

### Common Issues

**1. Import errors:**
```bash
# Make sure you're in the project directory
cd /Users/shayan/Unina/rl_projs/marl_lgv_4

# Reinstall dependencies
pip install -r requirements.txt
```

**2. Environment not initialized:**
- Go to Configuration page
- Click "Initialize Environment"

**3. Streamlit not starting:**
```bash
# Check if port is in use
lsof -ti:8501 | xargs kill -9

# Restart streamlit
streamlit run app.py
```

**4. CUDA/GPU issues:**
- The system works on CPU
- GPU is optional for faster training
- Models automatically detect available device

## Tips for Best Results

1. **Start Small**: Begin with smaller warehouse (15x15) and fewer LGVs (3-4)
2. **Incremental Training**: Train for 50k steps first, then increase
3. **Tune Rewards**: Adjust reward weights based on desired behavior
4. **Compare Multiple Runs**: Run evaluations multiple times for statistical significance
5. **Use Baselines**: Always compare RL agents with A* and Dijkstra baselines
6. **Visualize**: Use simulation page to understand agent behavior
7. **Save Models**: Always save trained models for later use

## Next Steps

1. **Experiment with Parameters**: Try different warehouse sizes and configurations
2. **Compare Algorithms**: Train all three RL algorithms and compare
3. **Optimize Rewards**: Fine-tune reward function for specific objectives
4. **Extended Training**: Train for longer (500k+ timesteps) for better performance
5. **Custom Scenarios**: Create custom warehouse layouts and test scenarios
6. **Analysis**: Use the Analysis page to understand learning dynamics

## Support

For questions or issues:
- Check the Research page in the dashboard for methodology
- Review example outputs in the Analysis page
- Consult the code documentation in `src/` directory

## License

University of Naples - RL Project 2026

---

**Ready to start? Run:**
```bash
streamlit run app.py
```

Or use the quick start script:
```bash
./run_app.sh
```
