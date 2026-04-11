from typing import TypedDict, List
from agentlightning.types import Dataset


class CuriosityTask(TypedDict):
    """
    A typed dictionary representing a mathematical research task
    fed into the Agent-Lightning reinforcement framework.
    """

    inspiration_seed: str
    target_metric: str


INSPIRATION_SEEDS = [
    "Gaussian Mixture Models with Online Mean-Variance Update",
    "Fractional Differencing with Stationarity Search (find_min_d)",
    "Adaptive Kalman Filter for Dynamically Adjusting Parameter Innovation (Q/R Tuning)",
    "State-Space Model for Non-Linear Trend Estimation (Local Level with Trend)",
    "Ornstein-Uhlenbeck Mean Reversion with Bayesian Alpha Update",
    "Principal Component Analysis (PCA) for Eigenspectrum Decay and Correlation Entropy",
    "Student-T probability mapping for heavy tails (StudentTRegime)",
]


def load_training_dataset() -> Dataset[CuriosityTask]:
    """
    Load the inspiration seeds as a formal Dataset for Agent Lightning Trainer.
    """
    tasks: List[CuriosityTask] = [
        {"inspiration_seed": seed, "target_metric": "Sharpe"}
        for seed in INSPIRATION_SEEDS
    ]
    return tasks
