import numpy as np
from typing import List, Dict


class RoanCombiner:
    """
    Institutional Alpha Combination Engine (The Roan Protocol).
    Phase 24.6: Return-Weighted Variant.

    This version optimizes weights based on individual performance (Mean/Var)
    WITHOUT cross-sectional demeaning, preserving legal market drift (Beta).
    """

    def __init__(self, signal_names: List[str], lookback: int = 60):
        self.names = signal_names
        self.N = len(signal_names)
        self.M = lookback

        self.return_history = {name: [] for name in signal_names}
        self.weights = {name: 1.0 / self.N for name in signal_names}

        # Buffer for conviction z-scoring
        self.conviction_history = []

    def update(self, realizations: Dict[str, float]):
        """
        Update signal history and perform Mean-Variance Optimization.
        """
        for name in self.names:
            self.return_history[name].append(realizations.get(name, 0.0))
            if len(self.return_history[name]) > self.M:
                self.return_history[name].pop(0)

        if len(self.return_history[self.names[0]]) < self.M:
            return

        # N x M Matrix
        R = np.array([self.return_history[name] for name in self.names])

        # Step 2: Time-series Demean (For variance calculation)
        mu = R.mean(axis=1, keepdims=True)
        var = R.var(axis=1, keepdims=True) + 1e-8

        # Phase 24.6: Mean-Variance Weighting
        # w_i = mu_i / var_i
        # This is the optimal weight for independent signals.
        raw_weights = mu.flatten() / var.flatten()

        # Step 11: Normalize to unit sum
        # We use absolute sum normalization to prevent extreme leverage
        abs_sum = np.sum(np.abs(raw_weights))
        if abs_sum > 1e-7:
            self.weights = {
                name: raw_weights[i] / abs_sum for i, name in enumerate(self.names)
            }
        else:
            self.weights = {name: 1.0 / self.N for name in self.names}

    def combine(self, current_signals: Dict[str, float]) -> Dict:
        """
        Returns combined mega_alpha and rolling Z-score.
        """
        mega_alpha = 0.0
        for name, value in current_signals.items():
            mega_alpha += value * self.weights.get(name, 1.0 / self.N)

        self.conviction_history.append(mega_alpha)
        if len(self.conviction_history) > self.M:
            self.conviction_history.pop(0)

        if len(self.conviction_history) >= self.M:
            hist = np.array(self.conviction_history)
            mu = hist.mean()
            std = hist.std() + 1e-8
            z = (mega_alpha - mu) / std
        else:
            z = 0.0

        return {"mega_alpha": mega_alpha, "z_score": z}
