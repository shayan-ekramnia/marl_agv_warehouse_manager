"""
Runner for baseline algorithms
"""
import numpy as np
from typing import Dict, List, Tuple, Any
import time

from .pathfinding import AStarPlanner, DijkstraPlanner, GreedyPlanner, RandomPlanner
from ..environment.warehouse_env import WarehouseEnv
from ..environment.entities import Position


class BaselineRunner:
    """Run and evaluate baseline algorithms"""

    def __init__(self, env: WarehouseEnv):
        self.env = env

    def run_algorithm(self, algorithm: str, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Run baseline algorithm for multiple episodes

        Args:
            algorithm: 'A_star', 'Dijkstra', 'Greedy', 'Random'
            num_episodes: Number of episodes to run

        Returns:
            Dictionary with performance metrics
        """

        # Get planner class
        planner_map = {
            'A_star': AStarPlanner,
            'Dijkstra': DijkstraPlanner,
            'Greedy': GreedyPlanner,
            'Random': RandomPlanner
        }

        if algorithm not in planner_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        planner_class = planner_map[algorithm]

        # Metrics storage
        episode_rewards = []
        episode_lengths = []
        total_distances = []
        completion_rates = []
        avg_delivery_times = []
        collision_counts = []
        planning_times = []

        for episode in range(num_episodes):
            observations, _ = self.env.reset()

            episode_reward = 0
            episode_length = 0
            done = False

            # Create planner with current grid
            planner = planner_class(self.env.grid)

            # Plan paths for all LGVs
            lgv_plans = self._plan_all_lgvs(planner)

            while not done and episode_length < 1000:
                # Execute planned actions
                actions = self._execute_plans(lgv_plans)

                # Step environment
                observations, rewards, dones, truncated, info = self.env.step(actions)

                # Update metrics
                episode_reward += sum(rewards.values())
                episode_length += 1

                # Update plans if needed (re-plan when LGV reaches goal or picks up pallet)
                self._update_plans(lgv_plans, planner)

                done = dones.get('__all__', False)

            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            total_distances.append(info['total_distance'])
            completion_rates.append(info['completion_rate'])
            avg_delivery_times.append(info['avg_delivery_time'])
            collision_counts.append(info['total_collisions'])

        # Compute statistics
        results = {
            'algorithm': algorithm,
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_distance': np.mean(total_distances),
            'mean_completion_rate': np.mean(completion_rates),
            'mean_delivery_time': np.mean(avg_delivery_times),
            'mean_collisions': np.mean(collision_counts),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }

        return results

    def _plan_all_lgvs(self, planner) -> Dict[int, List[Tuple[int, int]]]:
        """Plan paths for all LGVs"""
        plans = {}

        for lgv in self.env.lgvs:
            path = self._plan_lgv(lgv, planner)
            plans[lgv.id] = path if path else []

        return plans

    def _plan_lgv(self, lgv, planner) -> List[Tuple[int, int]]:
        """Plan path for a single LGV"""
        start = lgv.position.to_grid()

        # Determine goal based on LGV state
        if lgv.has_load():
            # Go to delivery location
            goal = lgv.current_load.destination.to_grid()
        else:
            # Find nearest available pallet
            nearest_pallet = None
            min_dist = float('inf')

            for pallet in self.env.pallets:
                if not pallet.picked_up and not pallet.delivered:
                    dist = lgv.position.distance_to(pallet.position)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_pallet = pallet

            if nearest_pallet:
                goal = nearest_pallet.position.to_grid()
            else:
                # No pallets, stay in place
                return [start]

        # Plan path
        path = planner.find_path(start, goal)
        return path if path else [start]

    def _execute_plans(self, plans: Dict[int, List[Tuple[int, int]]]) -> Dict[int, np.ndarray]:
        """Convert planned paths to actions"""
        actions = {}

        for lgv in self.env.lgvs:
            lgv_id = lgv.id
            plan = plans.get(lgv_id, [])

            if len(plan) <= 1:
                # No path or at goal, try to load/unload
                action = self._get_load_action(lgv)
            else:
                # Follow path
                current_pos = lgv.position.to_grid()
                next_pos = plan[1] if len(plan) > 1 else plan[0]

                # Convert position difference to action
                action = self._position_to_action(current_pos, next_pos, lgv)

                # Remove current position from plan
                if plan and plan[0] == current_pos:
                    plan.pop(0)

            actions[lgv_id] = action

        return actions

    def _position_to_action(self, current: Tuple[int, int], target: Tuple[int, int], lgv) -> np.ndarray:
        """Convert position movement to action"""
        dx = target[0] - current[0]
        dy = target[1] - current[1]

        # Calculate desired direction
        desired_direction = np.arctan2(dy, dx)

        # Calculate angle difference
        angle_diff = desired_direction - lgv.direction
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # Determine steering action (0: left, 1: slight left, 2: straight, 3: slight right, 4: right)
        if angle_diff < -0.3:
            steer = 0  # Turn left
        elif angle_diff < -0.1:
            steer = 1  # Slight left
        elif angle_diff > 0.3:
            steer = 4  # Turn right
        elif angle_diff > 0.1:
            steer = 3  # Slight right
        else:
            steer = 2  # Straight

        # Acceleration (2: no accel, 3: forward, 4: fast forward)
        accel = 3 if lgv.velocity < lgv.max_speed * 0.8 else 2

        # No loading action while moving
        load = 0

        # Don't wait
        wait = 0

        return np.array([accel, steer, load, wait])

    def _get_load_action(self, lgv) -> np.ndarray:
        """Get action for loading/unloading"""
        # Determine if should load or unload
        should_load = False

        if lgv.can_load():
            # Check if near pallet
            for pallet in self.env.pallets:
                if not pallet.picked_up and lgv.position.distance_to(pallet.position) < 1.5:
                    should_load = True
                    break

        elif lgv.has_load():
            # Check if at delivery location
            if lgv.position.distance_to(lgv.current_load.destination) < 1.5:
                should_load = True  # Will trigger unload

        if should_load:
            return np.array([2, 2, 1, 0])  # No accel, straight, load/unload, no wait
        else:
            return np.array([2, 2, 0, 1])  # Wait in place

    def _update_plans(self, plans: Dict[int, List[Tuple[int, int]]], planner):
        """Update plans when LGVs reach goals or pick up pallets"""
        for lgv in self.env.lgvs:
            plan = plans.get(lgv.id, [])

            # Re-plan if at goal or no valid plan
            if len(plan) <= 1:
                new_plan = self._plan_lgv(lgv, planner)
                plans[lgv.id] = new_plan if new_plan else []

    def compare_all_algorithms(self, num_episodes: int = 100) -> Dict[str, Dict[str, Any]]:
        """Compare all baseline algorithms"""
        algorithms = ['A_star', 'Dijkstra', 'Greedy', 'Random']
        results = {}

        for algo in algorithms:
            print(f"Running {algo}...")
            results[algo] = self.run_algorithm(algo, num_episodes)

        return results
