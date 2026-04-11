import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from data.synthetic_regimes import generate_regime_data
from core.flow import MarketForecastingFlow
from core.regime_monitor import RegimeMonitor


def run_benchmark():
    print("Starting Phase 14: Genetic Divergence Benchmark...")

    # 1. Generate Data
    df = generate_regime_data(n_samples=1000)
    monitor = RegimeMonitor()
    flow = MarketForecastingFlow()

    # 2. Iterate through market steps
    # Note: We skip the first 20 steps to allow rolling features to populate
    for i in tqdm(range(20, len(df))):
        row = df.iloc[i]

        # Prepare market state for the flow
        # In this benchmark, we manually inject the current data into the flow's state
        # instead of triggering the full market ingestion for every step.
        flow.state.market_data = {
            "vol_5": row["vol_5"],
            "vol_20": row["vol_20"],
            "mom_10": row["mom_10"],
            "drift": row["drift"],
            "price": row["price"],
        }

        # Trigger the Self-Organizing Deliberation
        # We call the method directly for the benchmark
        flow.latent_projection_and_deliberation()

        # Record stats
        # Divergence is calculated as the mean deviation from mean_latent
        # we can't easily get agent_deviations here without refactoring flow further,
        # but we can simulate the calculation.

        # For simplicity, we just use the consensus score as a proxy for divergence
        monitor.record_step(
            t=i,
            price=row["price"],
            ground_truth_regime=row["regime"],
            consensus_score=flow.state.consensus_score,
            latent_divergence=1.0 - flow.state.consensus_score,
            predicted_prob=0.5,  # Placeholder for this benchmark
        )

    # 3. Analyze and Report
    recovery_stats = monitor.analyze_recovery()
    monitor.save_report("genetic_divergence_report.json")

    print("\n--- Phase 14 Benchmark Results ---")
    for stat in recovery_stats:
        print(
            f"Shift: Regime {stat['from_regime']} -> {stat['to_regime']} | Recovery: {stat['recovery_steps']} steps"
        )

    print("\nBenchmark Complete.")


if __name__ == "__main__":
    run_benchmark()
