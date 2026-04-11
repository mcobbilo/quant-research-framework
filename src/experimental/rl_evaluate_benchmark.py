import os
import sys
import pandas as pd
import torch
import warnings
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

# Guarantee path context
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experimental.rl_environment import QuantMarketEnv


def calculate_drawdown(series):
    roll_max = series.cummax()
    return (series / roll_max) - 1.0


def evaluate_drone():
    print("Initializing RL Matrix Benchmark...")
    env = QuantMarketEnv()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "ppo_quant_drone.zip",
    )

    try:
        model = PPO.load(model_path, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    obs, _ = env.reset()
    records = []

    terminated = False

    while not terminated:
        date = env.df.loc[env.current_step, "Date"]
        spy_price = env.df.loc[env.current_step, "Close"]

        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, term, trunc, info = env.step(action)

        records.append(
            {"Date": date, "Drone_Balance": info["balance"], "SPY_Price": spy_price}
        )

        terminated = term

    print("Trace Complete. Calculating Matrix Physics...")

    results = pd.DataFrame(records)
    results["Date"] = pd.to_datetime(results["Date"])
    results.set_index("Date", inplace=True)

    initial_spy = results["SPY_Price"].iloc[0]
    initial_drone = 10000.0

    results["Drone_Cum"] = results["Drone_Balance"] / initial_drone
    results["SPY_Cum"] = results["SPY_Price"] / initial_spy

    results["Drone_DD"] = calculate_drawdown(results["Drone_Cum"])
    results["SPY_DD"] = calculate_drawdown(results["SPY_Cum"])

    results["Year"] = results.index.year

    def process_year(group):
        d_p = group["Drone_Cum"].iloc[-1] / group["Drone_Cum"].iloc[0] - 1.0
        s_p = group["SPY_Cum"].iloc[-1] / group["SPY_Cum"].iloc[0] - 1.0
        d_dd = group["Drone_DD"].min()
        s_dd = group["SPY_DD"].min()
        return pd.Series(
            {
                "Drone_Yr_Ret": d_p,
                "SPY_Yr_Ret": s_p,
                "Drone_Max_DD": d_dd,
                "SPY_Max_DD": s_dd,
            }
        )

    yearly = results.groupby("Year").apply(process_year)

    print("\n========================================================")
    print("10M ITERATION RL DRONE vs SPY BENCHMARK")
    print("========================================================")
    print(f"Total Drone Return: {results['Drone_Cum'].iloc[-1] - 1.0:.2%}")
    print(f"Total S&P 500 Return: {results['SPY_Cum'].iloc[-1] - 1.0:.2%}")
    print(f"Maximum Drone Drawdown: {results['Drone_DD'].min():.2%}")
    print(f"Maximum S&P 500 Drawdown: {results['SPY_DD'].min():.2%}")
    print("========================================================\n")

    print("YEAR | DRONE RET | SPY RET  | DRONE DD  | SPY DD")
    print("--------------------------------------------------")
    for year, row in yearly.iterrows():
        print(
            f"{int(year)} | {row['Drone_Yr_Ret']:>9.2%} | {row['SPY_Yr_Ret']:>8.2%} | {row['Drone_Max_DD']:>9.2%} | {row['SPY_Max_DD']:>8.2%}"
        )


if __name__ == "__main__":
    evaluate_drone()
