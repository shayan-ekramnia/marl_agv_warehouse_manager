"""
Base Agent class for all RL algorithms
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class BaseAgent(ABC):
    """Base class for RL agents"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None

    @abstractmethod
    def train(self, env, total_timesteps: int) -> Dict[str, Any]:
        """Train the agent"""
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        """Predict action given observation"""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model"""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model"""
        pass

    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics"""
        return {}
