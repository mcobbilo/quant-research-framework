import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.deliberation import LatentCouncil

def test_sequential_consensus():
    print("Testing Sequential Latent Deliberation...")
    feature_dim = 4
    latent_dim = 128
    council = LatentCouncil(feature_dim=feature_dim, latent_dim=latent_dim)
    
    features = torch.randn(1, feature_dim)
    council_size = 5
    agent_latents = []
    current_consensus = None
    
    for i in range(council_size):
        agent_latent = council.project_agent_reasoning(features, previous_consensus=current_consensus)
        agent_latents.append(agent_latent)
        current_consensus = torch.mean(torch.stack(agent_latents), dim=0)
        print(f"  Step {i+1}: Latent norm = {torch.norm(agent_latent).item():.4f}")

    consensus_data = council.calculate_consensus(agent_latents)
    print(f"Final Consensus Score: {consensus_data['consensus_score']:.4f}")
    assert len(agent_latents) == council_size
    assert consensus_data['mean_latent'].shape == (1, latent_dim)
    print("Verification Successful!")

if __name__ == "__main__":
    try:
        test_sequential_consensus()
    except Exception as e:
        print(f"Verification Failed: {e}")
        sys.exit(1)
