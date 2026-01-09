"""
Multi-Agent Warehouse Environment for RL
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import yaml

from .entities import LGV, Pallet, Shelf, Position, Order


class WarehouseEnv(gym.Env):
    """
    Multi-Agent Warehouse Environment

    Observation Space (per agent):
        - Agent position (2)
        - Agent velocity (1)
        - Agent direction (1)
        - Has load (1)
        - Nearest pallet position (2)
        - Nearest pallet distance (1)
        - Target position (2)
        - Other agents positions (2 * num_agents)
        - Grid occupancy around agent (5x5 grid)

    Action Space (per agent):
        - Move forward/backward
        - Turn left/right
        - Load/unload
        - Wait
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Warehouse parameters
        self.width = self.config['warehouse']['width']
        self.height = self.config['warehouse']['height']
        self.num_shelves = self.config['warehouse']['num_shelves']
        self.num_pallets = self.config['warehouse']['num_pallets']
        self.num_lgvs = self.config['warehouse']['num_lgvs']

        # LGV parameters
        lgv_config = self.config['lgv']
        self.max_speed = lgv_config['max_speed']
        self.max_acceleration = lgv_config['max_acceleration']
        self.turning_radius = lgv_config['turning_radius']
        self.loading_time = lgv_config['loading_time']
        self.unloading_time = lgv_config['unloading_time']

        # Reward weights
        self.rewards = self.config['rewards']

        # Define action and observation spaces
        # Action space: [acceleration, steering, load/unload, wait]
        self.action_space = spaces.MultiDiscrete([5, 5, 2, 2])  # Per agent

        # Observation space
        obs_dim = 35 + 2 * self.num_lgvs  # Base obs + other agents
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize environment
        self.lgvs: List[LGV] = []
        self.pallets: List[Pallet] = []
        self.shelves: List[Shelf] = []
        self.orders: List[Order] = []
        self.grid = None
        self.current_step = 0
        # Adaptive max_steps based on task complexity
        # Simple heuristic: ~50 steps per pallet + buffer
        self.max_steps = max(200, self.num_pallets * 50)
        self.current_time = 0.0
        self.dt = 1.0  # Time step in seconds

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Initialize grid (0: free, 1: shelf, 2: lgv, 3: pallet)
        self.grid = np.zeros((self.width, self.height), dtype=np.int32)

        # Create shelves
        self.shelves = self._create_shelves()
        for shelf in self.shelves:
            x, y = shelf.position.to_grid()
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[x, y] = 1

        # Create LGVs
        self.lgvs = self._create_lgvs()

        # Create pallets
        self.pallets = self._create_pallets()

        # Create orders
        self.orders = []

        self.current_step = 0
        self.current_time = 0.0

        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one time step

        Args:
            actions: Dictionary mapping agent_id to action array

        Returns:
            observations, rewards, dones, truncated, info
        """
        self.current_step += 1
        self.current_time += self.dt

        rewards = {i: 0.0 for i in range(self.num_lgvs)}

        # Execute actions for each LGV
        for lgv_id, action in actions.items():
            if lgv_id < len(self.lgvs):
                reward = self._execute_action(lgv_id, action)
                rewards[lgv_id] = reward

        # Update environment state
        self._update_grid()

        # Generate new pallets/orders randomly
        if np.random.random() < 0.1 and len(self.pallets) < self.num_pallets:
            self._add_random_pallet()

        # Get observations
        observations = self._get_observations()

        # Check if episode is done
        all_delivered = all(p.delivered for p in self.pallets)
        timeout = self.current_step >= self.max_steps

        dones = {i: all_delivered or timeout for i in range(self.num_lgvs)}
        dones['__all__'] = all_delivered or timeout

        truncated = {i: timeout for i in range(self.num_lgvs)}
        truncated['__all__'] = timeout

        info = self._get_info()

        return observations, rewards, dones, truncated, info

    def _create_shelves(self) -> List[Shelf]:
        """Create shelves in warehouse"""
        shelves = []

        # Create shelves in a grid pattern with aisles
        shelf_id = 0
        for i in range(3, self.width - 3, 4):
            for j in range(2, self.height - 2, 4):
                if shelf_id < self.num_shelves:
                    shelf = Shelf(
                        id=shelf_id,
                        position=Position(float(i), float(j)),
                        width=2.0,
                        height=2.0
                    )
                    shelves.append(shelf)
                    shelf_id += 1

        return shelves

    def _create_lgvs(self) -> List[LGV]:
        """Create LGVs at starting positions"""
        lgvs = []

        # Place LGVs at bottom of warehouse
        spacing = self.width / (self.num_lgvs + 1)
        for i in range(self.num_lgvs):
            x = spacing * (i + 1)
            y = 1.0

            lgv = LGV(
                id=i,
                position=Position(x, y),
                max_speed=self.max_speed,
                max_acceleration=self.max_acceleration,
                turning_radius=self.turning_radius
            )
            lgvs.append(lgv)

        return lgvs

    def _create_pallets(self) -> List[Pallet]:
        """Create pallets at random locations"""
        pallets = []

        for i in range(self.num_pallets):
            # Random pickup location (avoid shelves)
            while True:
                x = np.random.uniform(1, self.width - 1)
                y = np.random.uniform(1, self.height - 1)
                gx, gy = int(x), int(y)
                if self.grid[gx, gy] == 0:
                    break

            # Random delivery location
            while True:
                dx = np.random.uniform(1, self.width - 1)
                dy = np.random.uniform(1, self.height - 1)
                dgx, dgy = int(dx), int(dy)
                if self.grid[dgx, dgy] == 0 and (dx != x or dy != y):
                    break

            pallet = Pallet(
                id=i,
                position=Position(x, y),
                destination=Position(dx, dy),
                priority=np.random.randint(1, 4),
                creation_time=self.current_time
            )
            pallets.append(pallet)

        return pallets

    def _add_random_pallet(self):
        """Add a new random pallet"""
        pallet_id = len(self.pallets)

        # Random positions
        x = np.random.uniform(1, self.width - 1)
        y = np.random.uniform(1, self.height - 1)
        dx = np.random.uniform(1, self.width - 1)
        dy = np.random.uniform(1, self.height - 1)

        pallet = Pallet(
            id=pallet_id,
            position=Position(x, y),
            destination=Position(dx, dy),
            priority=np.random.randint(1, 4),
            creation_time=self.current_time
        )
        self.pallets.append(pallet)

    def _execute_action(self, lgv_id: int, action: np.ndarray) -> float:
        """Execute action for an LGV and return reward"""
        lgv = self.lgvs[lgv_id]
        reward = 0.0

        # Parse action
        accel_action = action[0] - 2  # [-2, -1, 0, 1, 2]
        steer_action = (action[1] - 2) * 0.2  # [-0.4, -0.2, 0, 0.2, 0.4] radians
        load_action = action[2]  # 0: no action, 1: load/unload
        wait_action = action[3]  # 0: move, 1: wait

        if wait_action == 1:
            # Waiting
            lgv.idle_time += self.dt
            reward += self.rewards['idle_penalty']
            return reward

        # Handle loading/unloading
        if lgv.is_loading or lgv.is_unloading:
            lgv.loading_timer += self.dt

            if lgv.is_loading and lgv.loading_timer >= self.loading_time:
                # Finish loading
                lgv.is_loading = False
                lgv.loading_timer = 0.0
                reward += 5.0  # Small bonus for loading

            elif lgv.is_unloading and lgv.loading_timer >= self.unloading_time:
                # Finish unloading
                if lgv.current_load:
                    lgv.current_load.delivered = True
                    lgv.current_load.delivery_time = self.current_time
                    lgv.total_deliveries += 1
                    reward += self.rewards['delivery_success']

                lgv.current_load = None
                lgv.is_unloading = False
                lgv.loading_timer = 0.0

            return reward

        # Try to load/unload if requested
        if load_action == 1:
            if lgv.can_load():
                # Try to pick up nearby pallet
                for pallet in self.pallets:
                    if not pallet.picked_up and pallet.position.distance_to(lgv.position) < 1.5:
                        lgv.current_load = pallet
                        lgv.is_loading = True
                        lgv.loading_timer = 0.0
                        pallet.picked_up = True
                        pallet.assigned_lgv = lgv_id
                        pallet.pickup_time = self.current_time
                        reward += self.rewards.get('pickup_success', 20.0)
                        break

            elif lgv.has_load():
                # Try to deliver at destination
                if lgv.position.distance_to(lgv.current_load.destination) < 1.5:
                    lgv.is_unloading = True
                    lgv.loading_timer = 0.0

        # Update velocity
        acceleration = accel_action * self.max_acceleration * self.dt
        lgv.velocity = np.clip(lgv.velocity + acceleration, 0, self.max_speed)

        # Update direction
        lgv.direction += steer_action
        lgv.direction = lgv.direction % (2 * np.pi)

        # Update position
        old_pos = Position(lgv.position.x, lgv.position.y)
        new_x = lgv.position.x + lgv.velocity * np.cos(lgv.direction) * self.dt
        new_y = lgv.position.y + lgv.velocity * np.sin(lgv.direction) * self.dt

        # Check bounds
        new_x = np.clip(new_x, 0, self.width - 1)
        new_y = np.clip(new_y, 0, self.height - 1)

        # Check collisions
        collision = self._check_collision(lgv_id, Position(new_x, new_y))

        if not collision:
            lgv.update_position(Position(new_x, new_y), self.dt)

            # Progress-based rewards (task-oriented)
            if lgv.has_load():
                # Reward for moving towards delivery destination
                old_dist = old_pos.distance_to(lgv.current_load.destination)
                new_dist = lgv.position.distance_to(lgv.current_load.destination)
                progress = old_dist - new_dist

                if progress > 0:
                    # Positive reward for making progress
                    reward += self.rewards.get('progress_to_delivery', 3.0) * progress
                elif progress < 0:
                    # Small penalty for moving away from goal
                    reward += progress * 0.5
            else:
                # Reward for moving towards nearest available pallet
                nearest_pallet = None
                min_dist = float('inf')
                for pallet in self.pallets:
                    if not pallet.picked_up and not pallet.delivered:
                        dist = lgv.position.distance_to(pallet.position)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_pallet = pallet

                if nearest_pallet:
                    old_dist = old_pos.distance_to(nearest_pallet.position)
                    new_dist = lgv.position.distance_to(nearest_pallet.position)
                    progress = old_dist - new_dist

                    if progress > 0:
                        # Positive reward for approaching pallet
                        reward += self.rewards.get('progress_to_pickup', 2.0) * progress
                    elif progress < 0:
                        # Small penalty for moving away
                        reward += progress * 0.3

        else:
            # Collision occurred
            lgv.collision_count += 1
            reward += self.rewards.get('collision_penalty', -5.0)

        # Small step penalty to encourage task completion
        reward += self.rewards.get('step_penalty', -0.02)

        return reward

    def _check_collision(self, lgv_id: int, new_position: Position) -> bool:
        """Check if position would cause collision"""
        gx, gy = new_position.to_grid()

        # Check bounds
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return True

        # Check shelf collision
        if self.grid[gx, gy] == 1:
            return True

        # Check collision with other LGVs
        for i, other_lgv in enumerate(self.lgvs):
            if i != lgv_id:
                if new_position.distance_to(other_lgv.position) < 1.0:
                    return True

        return False

    def _update_grid(self):
        """Update grid state"""
        # Reset dynamic objects
        self.grid = np.where(self.grid == 1, 1, 0)

        # Add LGVs
        for lgv in self.lgvs:
            gx, gy = lgv.position.to_grid()
            if 0 <= gx < self.width and 0 <= gy < self.height:
                self.grid[gx, gy] = 2

        # Add pallets
        for pallet in self.pallets:
            if not pallet.picked_up and not pallet.delivered:
                gx, gy = pallet.position.to_grid()
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    self.grid[gx, gy] = 3

    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents"""
        observations = {}

        for i, lgv in enumerate(self.lgvs):
            obs = self._get_agent_observation(i)
            observations[i] = obs

        return observations

    def _get_agent_observation(self, lgv_id: int) -> np.ndarray:
        """Get observation for a single agent"""
        lgv = self.lgvs[lgv_id]
        obs = []

        # Agent state (5)
        obs.extend([
            lgv.position.x / self.width,
            lgv.position.y / self.height,
            lgv.velocity / self.max_speed,
            lgv.direction / (2 * np.pi),
            1.0 if lgv.has_load() else 0.0
        ])

        # Nearest pallet (3)
        if lgv.has_load():
            # Target is delivery location
            target = lgv.current_load.destination
            obs.extend([
                target.x / self.width,
                target.y / self.height,
                lgv.position.distance_to(target) / (self.width + self.height)
            ])
        else:
            # Find nearest available pallet
            nearest_pallet = None
            min_dist = float('inf')

            for pallet in self.pallets:
                if not pallet.picked_up and not pallet.delivered:
                    dist = lgv.position.distance_to(pallet.position)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_pallet = pallet

            if nearest_pallet:
                obs.extend([
                    nearest_pallet.position.x / self.width,
                    nearest_pallet.position.y / self.height,
                    min_dist / (self.width + self.height)
                ])
            else:
                obs.extend([0.0, 0.0, 1.0])

        # Target position (2)
        if lgv.target_position:
            obs.extend([
                lgv.target_position.x / self.width,
                lgv.target_position.y / self.height
            ])
        else:
            obs.extend([0.0, 0.0])

        # Other agents (2 * num_lgvs)
        for other_lgv in self.lgvs:
            obs.extend([
                other_lgv.position.x / self.width,
                other_lgv.position.y / self.height
            ])

        # Local grid (5x5 = 25)
        gx, gy = lgv.position.to_grid()
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    obs.append(float(self.grid[nx, ny]) / 3.0)
                else:
                    obs.append(1.0)  # Treat out of bounds as obstacle

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict:
        """Get environment info"""
        total_deliveries = sum(lgv.total_deliveries for lgv in self.lgvs)
        total_distance = sum(lgv.total_distance for lgv in self.lgvs)
        total_collisions = sum(lgv.collision_count for lgv in self.lgvs)

        delivered_pallets = sum(1 for p in self.pallets if p.delivered)
        pending_pallets = sum(1 for p in self.pallets if not p.picked_up and not p.delivered)
        in_transit = sum(1 for p in self.pallets if p.picked_up and not p.delivered)

        avg_delivery_time = 0.0
        if delivered_pallets > 0:
            delivery_times = [p.delivery_time - p.creation_time for p in self.pallets if p.delivered and p.delivery_time]
            if delivery_times:
                avg_delivery_time = np.mean(delivery_times)

        return {
            'step': self.current_step,
            'time': self.current_time,
            'total_deliveries': total_deliveries,
            'total_distance': total_distance,
            'total_collisions': total_collisions,
            'delivered_pallets': delivered_pallets,
            'pending_pallets': pending_pallets,
            'in_transit_pallets': in_transit,
            'avg_delivery_time': avg_delivery_time,
            'completion_rate': delivered_pallets / len(self.pallets) if self.pallets else 0.0
        }

    def render(self, mode='human'):
        """Render the environment (used by visualization module)"""
        return self.grid.copy()

    def get_state(self) -> Dict:
        """Get complete environment state for visualization"""
        return {
            'lgvs': [
                {
                    'id': lgv.id,
                    'position': lgv.position.to_tuple(),
                    'direction': lgv.direction,
                    'velocity': lgv.velocity,
                    'has_load': lgv.has_load(),
                    'is_loading': lgv.is_loading,
                    'is_unloading': lgv.is_unloading,
                    'total_distance': lgv.total_distance,
                    'total_deliveries': lgv.total_deliveries,
                    'collision_count': lgv.collision_count
                }
                for lgv in self.lgvs
            ],
            'pallets': [
                {
                    'id': pallet.id,
                    'position': pallet.position.to_tuple(),
                    'destination': pallet.destination.to_tuple(),
                    'picked_up': pallet.picked_up,
                    'delivered': pallet.delivered,
                    'priority': pallet.priority,
                    'assigned_lgv': pallet.assigned_lgv
                }
                for pallet in self.pallets
            ],
            'shelves': [
                {
                    'id': shelf.id,
                    'position': shelf.position.to_tuple(),
                    'width': shelf.width,
                    'height': shelf.height
                }
                for shelf in self.shelves
            ],
            'grid': self.grid.copy(),
            'info': self._get_info()
        }
