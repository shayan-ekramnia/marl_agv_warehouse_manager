"""
Entity classes for warehouse simulation
"""
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class Position:
    """2D position in warehouse"""
    x: float
    y: float

    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_grid(self) -> Tuple[int, int]:
        """Convert to grid coordinates"""
        return (int(self.x), int(self.y))


@dataclass
class LGV:
    """Automated Guided Vehicle"""
    id: int
    position: Position
    velocity: float = 0.0
    direction: float = 0.0  # radians
    max_speed: float = 2.0
    max_acceleration: float = 0.5
    turning_radius: float = 1.0
    load_capacity: int = 1
    current_load: Optional['Pallet'] = None
    target_position: Optional[Position] = None
    is_loading: bool = False
    is_unloading: bool = False
    loading_timer: float = 0.0

    # Statistics
    total_distance: float = 0.0
    total_deliveries: int = 0
    idle_time: float = 0.0
    collision_count: int = 0

    def has_load(self) -> bool:
        """Check if LGV is carrying a pallet"""
        return self.current_load is not None

    def can_load(self) -> bool:
        """Check if LGV can load a pallet"""
        return not self.has_load() and not self.is_loading and not self.is_unloading

    def update_position(self, new_position: Position, dt: float = 1.0):
        """Update position and track distance"""
        dist = self.position.distance_to(new_position)
        self.total_distance += dist
        self.position = new_position


@dataclass
class Pallet:
    """Pallet that needs to be transported"""
    id: int
    position: Position
    destination: Position
    priority: int = 1
    assigned_lgv: Optional[int] = None
    picked_up: bool = False
    delivered: bool = False
    creation_time: float = 0.0
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None


@dataclass
class Shelf:
    """Storage shelf in warehouse"""
    id: int
    position: Position
    width: float = 1.0
    height: float = 2.0
    occupied_slots: int = 0
    max_slots: int = 10

    def is_full(self) -> bool:
        return self.occupied_slots >= self.max_slots


@dataclass
class Order:
    """Delivery order"""
    id: int
    pallets: List[int]
    pickup_locations: List[Position]
    delivery_location: Position
    deadline: Optional[float] = None
    completed: bool = False
    completion_time: Optional[float] = None
