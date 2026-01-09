"""
Baseline pathfinding algorithms for comparison
"""
import numpy as np
import heapq
from typing import List, Tuple, Optional, Set
from abc import ABC, abstractmethod

from ..environment.entities import Position, LGV, Pallet


class PathPlanner(ABC):
    """Base class for path planning algorithms"""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.height, self.width = grid.shape

    @abstractmethod
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find path from start to goal"""
        pass

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = []

        # 4-connectivity (up, down, left, right)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                # Allow movement on free space (0) and pallet locations (3)
                if self.grid[nx, ny] in [0, 3]:
                    neighbors.append((nx, ny))

        return neighbors


class AStarPlanner(PathPlanner):
    """A* pathfinding algorithm"""

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find optimal path using A* algorithm"""

        def heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance heuristic"""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Priority queue: (f_score, counter, position, path)
        counter = 0
        open_set = [(heuristic(start), counter, start, [start])]
        closed_set: Set[Tuple[int, int]] = set()
        g_scores = {start: 0}

        while open_set:
            f_score, _, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if current in closed_set:
                continue

            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_scores[current] + 1

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor, path + [neighbor]))

        return None  # No path found


class DijkstraPlanner(PathPlanner):
    """Dijkstra's algorithm"""

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find optimal path using Dijkstra's algorithm"""

        # Priority queue: (cost, counter, position, path)
        counter = 0
        open_set = [(0, counter, start, [start])]
        visited: Set[Tuple[int, int]] = set()
        costs = {start: 0}

        while open_set:
            cost, _, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if current in visited:
                continue

            visited.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue

                new_cost = cost + 1

                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    counter += 1
                    heapq.heappush(open_set, (new_cost, counter, neighbor, path + [neighbor]))

        return None  # No path found


class GreedyPlanner(PathPlanner):
    """Greedy best-first search"""

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find path using greedy best-first search"""

        def heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance heuristic"""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Priority queue: (heuristic, counter, position, path)
        counter = 0
        open_set = [(heuristic(start), counter, start, [start])]
        visited: Set[Tuple[int, int]] = set()

        while open_set:
            h, _, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if current in visited:
                continue

            visited.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    counter += 1
                    heapq.heappush(open_set, (heuristic(neighbor), counter, neighbor, path + [neighbor]))

        return None  # No path found


class RandomPlanner(PathPlanner):
    """Random walk planner (baseline)"""

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = 1000) -> Optional[List[Tuple[int, int]]]:
        """Find path using random walk with goal bias"""

        path = [start]
        current = start
        visited_count = {start: 0}

        for step in range(max_steps):
            if current == goal:
                return path

            neighbors = self.get_neighbors(current)
            if not neighbors:
                return None

            # Bias towards goal with 50% probability
            if np.random.random() < 0.5:
                # Move towards goal
                best_neighbor = min(neighbors,
                                  key=lambda n: abs(n[0] - goal[0]) + abs(n[1] - goal[1]))
                next_pos = best_neighbor
            else:
                # Random move, prefer unvisited
                unvisited = [n for n in neighbors if n not in visited_count]
                if unvisited:
                    next_pos = np.random.choice(len(unvisited))
                    next_pos = unvisited[next_pos]
                else:
                    # Visit least visited
                    next_pos = min(neighbors, key=lambda n: visited_count.get(n, 0))

            current = next_pos
            path.append(current)
            visited_count[current] = visited_count.get(current, 0) + 1

        return path if current == goal else None
