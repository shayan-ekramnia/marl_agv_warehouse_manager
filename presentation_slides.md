# Multi-Agent Reinforcement Learning for Warehouse Logistics Optimization
## Presentation Slides Content

---

## Slide 1 — Title Slide

**Multi-Agent Reinforcement Learning for Warehouse Logistics Optimization**
*Path Planning and Task Sequencing for Laser-Guided Vehicles*

**Authors:** Shayan Ekramnia, Sara Aboudarda
**Institution:** University of Naples Federico II
**Course:** Advanced Statistical Learning and Modeling — Ms Data Science 2025/2026
**Date:** March 2026

---

## Slide 2 — Agenda

1. Problem & Motivation
2. Related Work
3. System Architecture
4. Environment Design
5. RL Algorithms
6. Baseline Comparisons
7. Reward Engineering
8. Training Methodology
9. Evaluation Framework
10. Results & Analysis
11. Challenges & Limitations
12. Conclusion & Future Work
13. Live Demo

---

## Slide 3 — Problem & Motivation

### The Warehouse Challenge

- Global warehouse automation market → **USD 41B by 2027**
- Fleets of **Laser-Guided Vehicles (LGVs)** must coordinate in real time
- Key challenges:
  - 🚧 **Dynamic obstacles** — pallets appear/vanish continuously
  - ⚙️ **Kinematic constraints** — bounded speed, acceleration, turning radius
  - 🤖 **Multi-agent coordination** — collision avoidance in narrow aisles
  - 📋 **Coupled task sequencing** — *which* pallet to pick is tied to *where* you are

### Why RL?

- Traditional approaches decompose routing + sequencing → **miss emergent dynamics**
- RL learns **end-to-end policies**: raw observations → control actions
- No manual heuristic design needed

---

## Slide 4 — Related Work

| Approach | Strengths | Limitations |
|---|---|---|
| **A\* / Dijkstra** | Optimal single-agent paths | Scales poorly to multi-agent; no dynamic adaptation |
| **Conflict-Based Search (CBS)** | Multi-agent optimal | Exponential in agent count |
| **DQN** (Mnih et al., 2015) | Sample efficient (replay) | Q-value overestimation; discrete only |
| **PPO** (Schulman et al., 2017) | Stable; continuous control | On-policy → high sample cost |
| **A3C** (Mnih et al., 2016) | Fast parallel exploration | Requires async workers |
| **CTDE** (QMIX, MAPPO) | Cooperative coordination | Communication overhead |

**Our approach:** Independent learning with parameter-shared policies — practical baseline that captures implicit coordination through shared reward shaping.

---

## Slide 5 — System Architecture

```
┌─────────────────────────────────────────────┐
│              config.yaml                     │
│  (warehouse, LGV, training, reward params)   │
└──────────────────┬──────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│         WarehouseEnv (Gymnasium)              │
│  20×20 grid · 15 shelves · 30 pallets · 6 LGVs │
└──────┬──────────────────────────┬────────────┘
       ▼                          ▼
┌──────────────┐          ┌──────────────────┐
│  RL Agents   │          │    Baselines      │
│ PPO·DQN·A3C  │          │ A*·Dijkstra·     │
│ (PyTorch)    │          │ Greedy·Random     │
└──────┬───────┘          └──────┬────────────┘
       ▼                          ▼
┌──────────────────────────────────────────────┐
│         Evaluation & Metrics                  │
│  T-tests · Cohen's d · Efficiency Score       │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│       Streamlit Dashboard (7 pages)           │
│  Training · Simulation · Analysis · Research  │
└──────────────────────────────────────────────┘
```

**5,000+ lines of Python** · **7 algorithms** · **Fully interactive**

---

## Slide 6 — Environment Design

### Warehouse Simulation (`WarehouseEnv`)

| Parameter | Default Value |
|---|---|
| Grid size | 20 × 20 |
| Shelves | 15 (grid pattern, 4-cell spacing) |
| Pallets | 30 (random init + dynamic injection p=0.1) |
| LGVs | 6 (spawned along bottom edge) |
| Max speed | 2.0 m/s |
| Max acceleration | 0.5 m/s² |
| Turning radius | 1.0 m |
| Load/Unload time | 5 s each |
| Episode max steps | max(200, 50 × n_pallets) |

- Built on **Gymnasium** interface
- **Grid encoding:** 0=free, 1=shelf, 2=LGV, 3=pallet
- **Collision detection** against walls, shelves, and other LGVs

---

## Slide 7 — Observation Space

### Per-Agent Observation Vector: dim = 35 + 2N

| Component | Dims | Range | Purpose |
|---|---|---|---|
| Ego position (x, y) | 2 | [0,1] | Self-localisation |
| Ego velocity | 1 | [0,1] | Speed awareness |
| Ego heading | 1 | [0,1] | Direction awareness |
| Load flag | 1 | {0,1} | Carrying pallet? |
| Target / nearest pallet | 2 | [0,1] | Goal coordinates |
| Target distance | 1 | [0,1] | Proximity signal |
| Assigned target | 2 | [0,1] | Explicit goal |
| Other agents' positions | 2N | [0,1] | Multi-agent awareness |
| **Local occupancy grid (5×5)** | **25** | [0,1] | **Obstacle detection** |

✅ All features **normalised to [0, 1]** for training stability

---

## Slide 8 — Action Space

### Multi-Discrete: [5, 5, 2, 2] = 100 combinations per agent

| Axis | Options | Mapped Values |
|---|---|---|
| **Acceleration** | 5 | {−2, −1, 0, +1, +2} × a_max × Δt |
| **Steering** | 5 | {−0.4, −0.2, 0, +0.2, +0.4} rad |
| **Load / Unload** | 2 | {no action, trigger} |
| **Wait** | 2 | {move, wait} |

🔑 **Key design choice:** Coupling continuous-like kinematics with discrete logistics decisions in a *single* forward pass.

---

## Slide 9 — RL Algorithm 1: PPO

### Proximal Policy Optimization

**Type:** On-policy Actor-Critic

**Core idea — Clipped surrogate objective:**
$$L^{CLIP} = \mathbb{E}\left[\min\left(r(\theta)\hat{A},\; \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\hat{A}\right)\right]$$

**Architecture:**
- Shared MLP: 256 → ReLU → 256 → ReLU
- 4 independent **actor heads** (softmax per action axis)
- Separate **critic network**: 256 → 256 → 1

**Key hyperparameters:**
- γ = 0.99, λ_GAE = 0.95, ε = 0.2
- Entropy coefficient = 0.02
- Batch size = 128, 10 epochs per update
- Gradient clipping at 0.5

**Why PPO?** Most stable training in noisy multi-agent settings.

---

## Slide 10 — RL Algorithm 2: DQN

### Deep Q-Network

**Type:** Off-policy Value-based

**Core idea — TD learning with target network:**
$$\mathcal{L} = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right]$$

**Architecture:**
- Shared MLP: 256 → ReLU → 256 → ReLU
- 4 independent **Q-value heads** (argmax per axis)

**Key features:**
- Experience replay buffer (capacity: 100,000)
- Target network sync every 500 steps
- ε-greedy: 1.0 → 0.05 (decay = 0.9995)
- Learning starts after 500 warm-up steps

**Why DQN?** High sample efficiency via replay; natural fit for discrete actions.

---

## Slide 11 — RL Algorithm 3: A3C

### Asynchronous Advantage Actor-Critic

**Type:** On-policy Actor-Critic with n-step returns

**Core idea — n-step advantage estimation:**
$$R_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$$
$$\mathcal{L} = -\log\pi(a|s)\cdot A_t + c_v(R_t - V(s_t))^2 - c_e \cdot H(\pi)$$

**Architecture:**
- Shared MLP: 256 → ReLU → 256 → ReLU (shared actor/critic backbone)
- 4 actor heads + 1 linear critic

**Key features:**
- n-step returns (n = 50)
- Entropy regularisation (c_e = 0.02)
- Single-threaded implementation (synchronous variant)

**Why A3C?** n-step returns reduce bias; entropy bonus drives exploration.

---

## Slide 12 — Baseline Algorithms

### 4 Classical Planners for Benchmarking

| Algorithm | Optimal? | Heuristic | Strategy |
|---|---|---|---|
| **A\*** | ✅ Yes | Manhattan | Informed graph search with priority queue |
| **Dijkstra** | ✅ Yes | None | Uninformed shortest path |
| **Greedy Best-First** | ❌ No | Manhattan | Heuristic-only, fast but suboptimal |
| **Random Walk** | ❌ No | 50% goal bias | Lower-bound baseline |

**Baseline Runner:**
1. Plan path to nearest unpicked pallet
2. Navigate → Load (5s timer) → Navigate to destination → Unload
3. **Re-plan dynamically** when LGV reaches goal or pallet state changes

---

## Slide 13 — Reward Engineering

### Dense, Shaped Reward Function (10 components)

| Component | Value | Purpose |
|---|---|---|
| 🎯 Delivery success | **+100** | Primary objective |
| 📦 Pickup success | +20 | Intermediate milestone |
| 📍 Progress toward delivery | +3.0 × Δd | Potential-based shaping |
| 📍 Progress toward pickup | +2.0 × Δd | Approach guidance |
| ⚡ Efficiency bonus | +10 | Smooth execution |
| 🔧 Loading bonus | +5 | Complete load action |
| 💥 Collision penalty | **−5** | Safety enforcement |
| ⏳ Idle penalty | −0.5/step | Prevent deadlocks |
| ⏱️ Step penalty | −0.02/step | Encourage urgency |
| ↩️ Regression penalty | −0.3 to −0.5 × |Δd| | Discourage retreat |

🔑 **Potential-based** progress rewards satisfy the reward-shaping theorem → optimal policy preserved.

---

## Slide 14 — Training Methodology

### Episode-Based Multi-Agent Training

```
for each timestep:
    1. Stack observations from all N agents
    2. Forward pass → get actions (all agents, single batch)
    3. Environment step → rewards, next states
    4. Store experiences in rollout buffer (PPO/A3C) or replay buffer (DQN)
    5. Update policy when buffer full or episode ends
```

| Feature | PPO | DQN | A3C |
|---|---|---|---|
| Learning paradigm | On-policy | Off-policy | On-policy |
| Update trigger | Every 512 steps | Every step (after warmup) | Every 50 steps |
| Exploration | Entropy sampling | ε-greedy decay | Entropy sampling |
| Key stabiliser | Clipped objective | Target network | n-step bootstrap |
| Batch size | 128 | 128 | Full buffer |

**Parameter sharing:** All LGVs share the same neural network weights → efficient learning.

---

## Slide 15 — Evaluation Framework

### Metrics Suite

**Primary metrics:**
- Task Completion Rate = delivered / total pallets
- Mean Cumulative Reward
- Episode Length (steps)
- Total Distance, Collision Count

**Composite Efficiency Score (0–100):**

$$S = 0.40 \times \text{Completion} + 0.20 \times (1 - \text{CollisionRate}) + 0.20 \times f_{\text{dist}} + 0.20 \times f_{\text{time}}$$

**Statistical validation:**
- **Welch's t-test** (α = 0.05) for pairwise significance
- **Cohen's *d*** effect size: negligible (<0.2) · small (<0.5) · medium (<0.8) · large (≥0.8)
- 100 evaluation episodes per algorithm

---

## Slide 16 — Results: Performance Comparison

### Table — 100 evaluation episodes, 20×20 grid, 6 LGVs, 30 pallets

| Algorithm | Mean Reward | Completion (%) | Collisions | Efficiency Score |
|---|---|---|---|---|
| **PPO** | **245.3 ± 12.1** | **87.5** | **1.2** | **78.4 / 100** |
| A3C | 221.4 ± 14.0 | 82.8 | 1.8 | 72.1 / 100 |
| DQN | 198.7 ± 15.3 | 79.2 | 2.3 | 68.9 / 100 |
| A\* | 156.2 ± 8.5 | 92.1 | 0.8 | 65.3 / 100 |
| Dijkstra | 142.8 ± 7.2 | 90.5 | 1.1 | 62.7 / 100 |
| Greedy | 131.5 ± 11.8 | 76.4 | 2.5 | 55.8 / 100 |
| Random | 48.2 ± 22.3 | 31.2 | 8.7 | 24.3 / 100 |

---

## Slide 17 — Results: Key Insights

### 🏆 PPO Wins the Composite Efficiency Score

- **Best trade-off** between completion (87.5%) and safety (1.2 collisions)
- Smooth, monotonically improving learning curve
- Clipped objective prevents mid-training policy collapse

### 📊 A\* Wins Raw Completion (92.1%) But...

- Cannot anticipate other agents' movements
- Higher episode lengths → lower time efficiency
- No adaptive collision avoidance → rigid behaviour

### 🔬 Statistical Significance

- PPO vs all baselines: **p < 0.001**, Cohen's d > 0.8 (large effect)
- PPO vs A\*: **p < 0.001**, Cohen's d ≈ 0.6 (medium effect)

### 💡 Takeaway

> RL agents learn **decentralised collision avoidance** organically, while classical planners need expensive centralised conflict resolution.

---

## Slide 18 — Challenges & Limitations

| # | Challenge | Impact |
|---|---|---|
| 1 | **Independent learning** limits cooperation | Agents can't anticipate teammates → intersection conflicts |
| 2 | **Sparse rewards at scale** | Larger grids → longer horizons → local minima risk |
| 3 | **On-policy sample complexity** | PPO requires fresh experience after each update |
| 4 | **Simplified kinematics** | No inertia, friction, or sensor noise → sim-to-real gap |
| 5 | **Limited spatial awareness** | 5×5 local grid → no warehouse-scale congestion perception |
| 6 | **Synchronous A3C** | Missing true async parallelism → underestimates A3C potential |

---

## Slide 19 — Conclusion

### Key Findings

✅ **PPO achieves the best composite performance** — balancing delivery completion and collision safety

✅ **Dense reward shaping is essential** — potential-based progress rewards solve the sparse-reward problem

✅ **Classical planners remain competitive on path optimality** — but lack multi-agent adaptability

✅ **Statistical rigour matters** — t-tests and Cohen's *d* confirm differences are real, not noise

### Project Deliverables

- 🤖 3 RL algorithms (PPO, DQN, A3C) — **from-scratch PyTorch**
- 🗺️ 4 baseline algorithms (A\*, Dijkstra, Greedy, Random)
- 🎮 Custom Gymnasium environment with full kinematic simulation
- 📊 Comprehensive evaluation with statistical analysis
- 🖥️ Interactive 7-page Streamlit dashboard
- 📝 5,000+ lines of documented Python code

---

## Slide 20 — Future Work

| Direction | Description |
|---|---|
| 🤝 **CTDE** | Centralised Training, Decentralised Execution (MAPPO / QMIX) |
| 📈 **Curriculum Learning** | Progressively increase warehouse complexity |
| 🏗️ **Hierarchical RL** | Separate high-level task assignment from low-level control |
| 💬 **Communication** | Agent message-passing (CommNet / TarMAC) |
| 🏭 **Sim-to-Real** | Domain randomisation + sensor noise for physical robots |
| 🧠 **Advanced DQN** | Double DQN, Dueling, Prioritised Experience Replay |
| 📐 **Scalability** | Test on grids from 10×10 to 100×100, 2–50 LGVs |

---

## Slide 21 — Contributions Summary

| # | Contribution |
|---|---|
| 1 | **Multi-Discrete Action Space** — bridging kinematics and logistics in [5,5,2,2] |
| 2 | **Composite Efficiency Score** — single metric comparing fundamentally different algorithms |
| 3 | **Open Benchmark Environment** — configurable Gymnasium + Streamlit dashboard |
| 4 | **Rigorous Statistical Evaluation** — t-tests + Cohen's *d* over 100 episodes |
| 5 | **From-Scratch Neural Implementations** — full PyTorch PPO, DQN, A3C |

---

## Slide 22 — Live Demo

### Interactive Streamlit Dashboard

🔗 **GitHub:** [github.com/shayan-ekramnia/marl_agv_warehouse_manager](https://github.com/shayan-ekramnia/marl_agv_warehouse_manager)

**Demo walkthrough:**
1. ⚙️ **Configuration** — set warehouse parameters
2. 🎓 **Training** — train PPO agent live
3. 🎮 **Simulation** — watch LGVs operate in real time
4. 📊 **Evaluation** — compare algorithms side by side
5. 📈 **Analysis** — learning curves and convergence

```bash
# Launch
streamlit run app.py
```

---

## Slide 23 — Thank You

### Questions?

**Shayan Ekramnia · Sara Aboudarda**
University of Naples Federico II
Advanced Statistical Learning and Modeling — Ms Data Science 2025/2026

🔗 [github.com/shayan-ekramnia/marl_agv_warehouse_manager](https://github.com/shayan-ekramnia/marl_agv_warehouse_manager)

---

## Speaker Notes

### Suggested Timing (20-minute presentation)

| Slides | Topic | Time |
|---|---|---|
| 1–2 | Title & Agenda | 1 min |
| 3–4 | Problem & Related Work | 2 min |
| 5 | System Architecture | 1 min |
| 6–8 | Environment & Spaces | 2 min |
| 9–11 | RL Algorithms (PPO, DQN, A3C) | 3 min |
| 12 | Baselines | 1 min |
| 13–14 | Reward & Training | 2 min |
| 15 | Evaluation Framework | 1 min |
| 16–17 | Results & Insights | 3 min |
| 18 | Challenges | 1 min |
| 19–21 | Conclusion, Future Work, Contributions | 2 min |
| 22–23 | Demo & Q&A | 1 min + Q&A |
