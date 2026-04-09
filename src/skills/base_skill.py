from abc import ABC, abstractmethod
import torch
import numpy as np

class JepaSkill(ABC):
    """
    Abstract Base Class for J-EPA Skills.
    Skills are modular mathematical operations ('Indicators', 'Regimes', 'Risk')
    that can be invoked by the Autonomous Curiosity Engine.
    """
    @abstractmethod
    def execute(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Primary execution logic for the skill. """
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        """ The J-EPA Attention Domain this skill contributes to. """
        pass

class VolTargetingSkill(JepaSkill):
    """
    Skill: Dynamic Volatility Targeting.
    Scales exposure based on rolling ATR and VIX regimes.
    """
    def __init__(self, target_vol: float = 0.10):
        self.target_vol = target_vol

    def execute(self, inputs: torch.Tensor) -> torch.Tensor:
        # Simple example: Inverse Volatility Scaling
        # (Batch, 1) -> (Batch, 1)
        vol = inputs.std(dim=1, keepdim=True) + 1e-6
        scaling_factor = self.target_vol / vol
        return scaling_factor

    @property
    def domain(self) -> str:
        return "VOLATILITY"

class GMMRegimeSkill(JepaSkill):
    """
    Skill: Gaussian Mixture Regime Classifier.
    Assigns soft-likelihoods for 5 distinct market regimes.
    """
    def execute(self, inputs: torch.Tensor) -> torch.Tensor:
        # J-EPA attention-biased regime clustering
        # (Batch, 16) -> (Batch, 5)
        return torch.softmax(inputs[:, :5], dim=-1)

    @property
    def domain(self) -> str:
        return "REGIME"
