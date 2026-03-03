# MARL Warehouse LGV Optimization - Final Report

**Institution**: University of Naples
**Project Track**: RL For Logistic Optimization (Track 1 & 2)
**Year**: 2026

## 1. Executive Summary
This project presents a complete, end-to-end Multi-Agent Reinforcement Learning (MARL) system designed to optimize Automated Guided Vehicle (LGV) operations in a dynamic warehouse environment. The core objective is to improve path planning and task sequencing for multiple autonomous agents, minimizing processing time and maximizing resource utilization. The final deliverable is a comprehensive, interactive application that unifies environment simulation, model training, statistical evaluation, and real-time visualization.

## 2. Environment & Digital Twin
The foundation of the project is a custom **Gymnasium-based warehouse digital twin** that accurately models:
- Dynamic grid layouts with configurable dimensions, shelves, pick-up/drop-off zones, and obstacles.
- Realistic LGV physics, incorporating kinematic constraints, acceleration, and turning geometry.
- Conflict resolution mechanisms, including strict collision detection and avoidance.
- Complex payload management (loading/unloading operations) and dynamic pallet generation.

## 3. Algorithmic Implementation
To benchmark and optimize LGV control, the system implements state-of-the-art RL algorithms alongside traditional pathfinding baselines via a modular architecture:
- **Reinforcement Learning (RL) Agents**: 
  - **PPO (Proximal Policy Optimization)**: Utilizes an Actor-Critic architecture with a clipped surrogate objective for stable training.
  - **DQN (Deep Q-Network)**: Leverages experience replay and epsilon-greedy exploration for discrete action spaces.
  - **A3C (Asynchronous Advantage Actor-Critic)**: Implements asynchronous multi-threading for robust exploration.
- **Baseline Pathfinding**: **A*** (optimal heuristic-based routing), **Dijkstra's Algortihm**, and **Greedy Best-First Search** are fully integrated to establish rigorous benchmarks against the trained RL policies. 

## 4. State Representation & Reward Design
The intelligence of the RL agents is driven by carefully engineered features:
- **Observation Space**: Normalized local grid occupancy (5x5), relative positions of other agents, payload status, and nearest target vectors.
- **Action Space**: A multi-discrete action space controlling acceleration, steering (radians), loading/unloading, and wait states.
- **Reward Function**: A shaped, multi-objective reward incorporating successful delivery bonuses (+100), efficiency incentives, distance penalties, and severe collision penalties (-50) to enforce safety constraints.

## 5. Evaluation & Visualization Framework
A sophisticated evaluation pipeline assesses LGV performance using a holistic framework:
- **Key Metrics**: Task completion rate, total distance traveled, collision frequency, average delivery time, and a normalized composite **Efficiency Score**.
- **Statistical Rigor**: Model comparative results are validated using T-tests and confidence intervals to ensure significance.
- **Streamlit Dashboard**: A professional 7-page web application provides an interactive interface for the system. It features live frame-by-frame simulation replay, trajectory tracking, configuration tweaking, training supervision, and statistical plotting (radar charts, learning curves, heatmaps).

## 6. Conclusion
The project successfully addresses all prescribed objectives. It demonstrates that Multi-Agent Reinforcement Learning can effectively solve complex, multi-agent logistics challenges by learning safe and optimal routing sequences. By pairing advanced AI models with a feature-rich digital twin and interactive visualizations, the system provides both a powerful optimization tool and an intuitive platform for further logistical research.
