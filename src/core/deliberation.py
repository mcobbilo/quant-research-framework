import os
import torch
import torch.nn as nn
from typing import List, Dict
from models.calm_engine import MarketAutoEncoder, BrierLoss, ActionDecoder

class EndogenousRoleAdapter(nn.Module):
    """
    Allows an agent to dynamically adapt its 'role' (projection bias) 
    based on the current market state and previous council consensus.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh() # Normalized direction shift
        )

    def forward(self, agent_latent: torch.Tensor, previous_consensus: torch.Tensor) -> torch.Tensor:
        # Combine current thought with existing council consensus
        combined = torch.cat([agent_latent, previous_consensus], dim=-1)
        shift = self.adapter(combined)
        return agent_latent + shift

from core.akf import AdaptiveKalmanFilter

class LatentCouncil:
    """
    Implements a High-Bandwidth Deliberation Council using CALM principles.
    Supports Self-Organizing Sequential Protocols (arXiv:2603.28990).
    Integrates Adaptive Kalman Filtering (AKF) for Phase 14 Genetic Divergence.
    """
    def __init__(self, feature_dim: int, latent_dim: int = 128, weights_path: str = "models/council_weights.pth"):
        self.model = MarketAutoEncoder(input_dim=feature_dim, latent_dim=latent_dim)
        self.role_adapter = EndogenousRoleAdapter(latent_dim=latent_dim)
        self.action_decoder = ActionDecoder(latent_dim=latent_dim) # New Agentic Action Head
        self.akf = AdaptiveKalmanFilter() # AKF for innovation tracking
        self.loss_fn = BrierLoss()
        self.latent_dim = latent_dim
        self.weights_path = weights_path

        # Phase 16: Auto-load existing calibration if available
        self.load_weights()

    def save_weights(self, path: str = None):
        """Saves current ActionDecoder and AutoEncoder weights."""
        target_path = path or self.weights_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'role_adapter_state_dict': self.role_adapter.state_dict(),
            'action_decoder_state_dict': self.action_decoder.state_dict(),
        }, target_path)
        print(f"[Council] Weights archived to {target_path}")

    def load_weights(self, path: str = None):
        """Loads ActionDecoder and AutoEncoder weights."""
        target_path = path or self.weights_path
        if os.path.exists(target_path):
            checkpoint = torch.load(target_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.role_adapter.load_state_dict(checkpoint['role_adapter_state_dict'])
            self.action_decoder.load_state_dict(checkpoint['action_decoder_state_dict'])
            print(f"[Council] Weights restored from {target_path}")
        else:
            print(f"[Council] No calibration weights found at {target_path}. Using random initialization.")

    def project_agent_reasoning(self, agent_features: torch.Tensor, previous_consensus: torch.Tensor = None) -> torch.Tensor:
        """
        Projects reasoning and optionally adapts it to the existing consensus (Sequential Protocol).
        Updates the AKF using the first feature dimension (e.g. price proxy) for innovation tracking.
        """
        # Phase 14: Update AKF with a scalar proxy from features (e.g., first feature)
        # This tracks 'innovation' (regime shifts)
        feature_proxy = agent_features[0, 0].item() if agent_features.dim() > 1 else agent_features[0].item()
        _, akf_q, innovation_score = self.akf.update(feature_proxy)
        
        with torch.no_grad():
            latent_vector = self.model.encode(agent_features)
        
        if previous_consensus is not None:
            # Self-organize by adapting to the current council direction
            # Innovation score could eventually scale the adapter learning rate or noise
            latent_vector = self.role_adapter(latent_vector, previous_consensus)
            
        return latent_vector

    def calculate_consensus(self, agent_latents: List[torch.Tensor]) -> Dict:
        """
        Calculates consensus using the Brier-like 'Collision' metric.
        """
        if not agent_latents:
            return {"consensus_score": 0.0, "mean_latent": torch.zeros((1, self.latent_dim))}
            
        stacked = torch.stack(agent_latents) 
        mean_latent = torch.mean(stacked, dim=0)
        
        similarities = []
        for i in range(len(agent_latents)):
            for j in range(i + 1, len(agent_latents)):
                sim = torch.nn.functional.cosine_similarity(agent_latents[i], agent_latents[j])
                similarities.append(sim.mean().item())
        
        avg_consensus = sum(similarities) / len(similarities) if similarities else 1.0
        
        return {
            "consensus_score": avg_consensus,
            "mean_latent": mean_latent,
            "agent_deviations": [torch.norm(l - mean_latent).item() for l in agent_latents]
        }

    def verify_prediction(self, predicted_latent: torch.Tensor, actual_market_features: torch.Tensor) -> float:
        """
        Likelihood-free verification of the predicted state vs actual market realization.
        Returns a 'Curiosity Signal' (Brier Score).
        """
        with torch.no_grad():
            actual_latent = self.model.encode(actual_market_features)
        
        # Brier Score / MSE in latent space
        drift = torch.nn.functional.mse_loss(predicted_latent, actual_latent).item()
        return drift

    def derive_action(self, latent_vector: torch.Tensor, consensus_score: float, vix: float = 20.0) -> str:
        """
        Maps a collective council 'thought' into a concrete flow action.
        Adaptive Threshold for Phase 23.2 Alpha Recovery.
        """
        # 1. Adaptive Uncertainty Guard
        # In low-vol (VIX < 15), we accept lower consensus to capture structural drift.
        threshold = 0.55 if vix < 15 else 0.65
        
        if consensus_score < threshold:
            return "trigger_human_review"
            
        # 2. Latent Action Decoding
        with torch.no_grad():
            action_logits = self.action_decoder(latent_vector)
            
            # Phase 23.1: Apply confidence-weighted bias to Long (Trade) to capture drift 
            if consensus_score > 0.85:
                action_logits[0, 1] += 0.5 
            
            action_idx = torch.argmax(action_logits, dim=-1).item()
            
        # Mapping: 0: Neutral, 1: Long (Trade), 2: Short (Hedge)
        mapping = {
            0: "trigger_human_review", # Neutral/Indeterminate
            1: "stage_trade",
            2: "stage_hedge"
        }
        
        return mapping.get(action_idx, "trigger_human_review")
