import os
from stable_baselines3 import PPO
from rl_environment import QuantMarketEnv
import warnings

warnings.filterwarnings("ignore")  # Suppress gym deprecation warnings


def evaluate_drone():
    model_path = os.path.join(os.path.dirname(__file__), "ppo_quant_drone.zip")

    print("==================================================")
    print("[*] Loading Trained Mathematical Matrix Weights...")
    print(f"[*] Booting {model_path}")
    print("==================================================")

    env = QuantMarketEnv()

    # Load the highly-trained weights
    model = PPO.load(model_path, env=env)

    obs, info = env.reset()
    terminated = False
    truncated = False

    print("\n[!] Initiating Full Lifetime Market Matrix Evaluation...")
    print("[!] Initial Capital: $10000.00")

    step_count = 0
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        step_count += 1

    print(f"\n---> Evaluation Complete! Analyzed {step_count} explicit market days.")
    print(f"---> AI Drone Lifetime Final Account Balance: ${info['balance']:.2f}")

    # Calculate CAGR
    years = step_count / 252
    total_return = info["balance"] / 10000.0
    if years > 0:
        cagr = (total_return ** (1 / years)) - 1
        print(f"---> Absolute Mathematical CAGR: {cagr * 100:.2f}%\n")


if __name__ == "__main__":
    evaluate_drone()
