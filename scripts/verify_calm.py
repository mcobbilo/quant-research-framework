import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.calm_engine import MarketAutoEncoder
from core.deliberation import LatentCouncil


def test_autoencoder_fidelity():
    print("\n--- Testing MarketAutoEncoder Fidelity ---")
    input_dim = 4
    latent_dim = 128
    model = MarketAutoEncoder(input_dim=input_dim, latent_dim=latent_dim)

    # Random market feature set [batch_size, input_dim]
    dummy_input = torch.randn(1, input_dim)

    recon, latent = model(dummy_input)

    mse = torch.nn.functional.mse_loss(recon, dummy_input).item()
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Latent Vector Shape: {latent.shape}")

    assert latent.shape == (1, latent_dim)
    print("AutoEncoder Test: PASSED")


def test_latent_council_consensus():
    print("\n--- Testing LatentCouncil Consensus ---")
    input_dim = 4
    council = LatentCouncil(feature_dim=input_dim)

    # 3 Agents with slightly different views
    alpha_view = torch.tensor([[0.2, 0.5, 0.8, 0.1]])
    beta_view = torch.tensor([[0.21, 0.49, 0.82, 0.09]])
    gamma_view = torch.tensor([[0.9, 0.1, 0.0, 0.5]])  # Dissident

    alpha_latent = council.project_agent_reasoning(alpha_view)
    beta_latent = council.project_agent_reasoning(beta_view)
    gamma_latent = council.project_agent_reasoning(gamma_view)

    # Check consensus between similar views
    consensus_ab = council.calculate_consensus([alpha_latent, beta_latent])
    print(f"Consensus (Alpha-Beta): {consensus_ab['consensus_score']:.4f}")

    # Check consensus with a dissident
    consensus_all = council.calculate_consensus(
        [alpha_latent, beta_latent, gamma_latent]
    )
    print(f"Consensus (Alpha-Beta-Gamma): {consensus_all['consensus_score']:.4f}")

    assert consensus_all["consensus_score"] < consensus_ab["consensus_score"]
    print("LatentCouncil Consensus Test: PASSED")


def test_curiosity_signal():
    print("\n--- Testing CALM Curiosity Signal (Brier) ---")
    council = LatentCouncil(feature_dim=4)

    market_now = torch.tensor([[0.2, 0.5, 0.8, 0.1]])
    predicted_latent = council.project_agent_reasoning(market_now)

    # Realization is exactly as expected
    market_future_stable = torch.tensor([[0.2, 0.5, 0.8, 0.1]])
    curiosity_stable = council.verify_prediction(predicted_latent, market_future_stable)
    print(f"Curiosity (Stable): {curiosity_stable:.6f}")

    # Realization is a major crash (reversal)
    market_future_crash = torch.tensor([[0.9, 0.1, 0.0, 0.5]])
    curiosity_crash = council.verify_prediction(predicted_latent, market_future_crash)
    print(f"Curiosity (Crash): {curiosity_crash:.6f}")

    assert curiosity_crash > curiosity_stable
    print("Curiosity Signal Test: PASSED")


if __name__ == "__main__":
    try:
        test_autoencoder_fidelity()
        test_latent_council_consensus()
        test_curiosity_signal()
        print("\nALL VERIFICATION TESTS PASSED.")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        sys.exit(1)
