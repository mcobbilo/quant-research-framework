import json
import statistics
from typing import List, Dict


class HarnessEvaluator:
    """
    Benchmarks the 'Yield per Prompt' for the J-EPA Curiosity Engine.
    Uses aggregate Sharpe Ratio as a metric for prompt hill-climbing.
    """

    def __init__(self, config_path: str = "src/core/harness_config.json"):
        self.config_path = config_path

    def calculate_harness_score(self, sharpes: List[float]) -> float:
        """
        Calculates a robust performance score for a harness configuration.
        Score = Mean Sharpe - (StdDev * 0.5) [Risk-Adjusted Alpha Potential]
        """
        if not sharpes:
            return 0.0

        mean_s = statistics.mean(sharpes)
        if len(sharpes) > 1:
            std_s = statistics.stdev(sharpes)
        else:
            std_s = 0.0

        return mean_s - (std_s * 0.5)

    def log_iteration(self, iter_id: int, configs: Dict, score: float):
        """Logs the iteration to harness_iterations.log for tracking."""
        with open("harness_iterations.log", "a") as f:
            log_entry = {
                "iteration": iter_id,
                "score": score,
                "config_snapshot": configs,
            }
            f.write(json.dumps(log_entry) + "\n")


if __name__ == "__main__":
    evaluator = HarnessEvaluator()
    # Mock data for initial validation
    mock_sharpes = [1.1, 1.3, 1.25]
    score = evaluator.calculate_harness_score(mock_sharpes)
    print(f"[HARNESS] Benchmark Score: {score:.4f}")
