import sys
import os

# Add src to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from core.flow import MarketForecastingFlow


def main():
    print("Initializing Autonomous Quantitative Research Framework (v1 Lean)...")
    flow = MarketForecastingFlow()
    result = flow.kickoff()
    print("Framework execution completed.")
    print(f"Final Output: {result}")


if __name__ == "__main__":
    main()
