import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return x_normed * self.weight

class AELayer(nn.Module):
    """
    Residual MLP Layer inspired by CALM architecture.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        return residual + self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

class MarketAutoEncoder(nn.Module):
    """
    Continuous AutoEncoder for high-bandwidth market state representation.
    Compresses high-dimensional technical and OSINT features into a latent vector.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128, n_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        curr_dim = input_dim
        for _ in range(n_layers):
            encoder_layers.append(AELayer(curr_dim, curr_dim * 2))
        self.encoder_backbone = nn.Sequential(*encoder_layers)
        self.latent_proj = nn.Linear(input_dim, latent_dim)
        
        # Decoder
        self.latent_expand = nn.Linear(latent_dim, input_dim)
        decoder_layers = []
        for _ in range(n_layers):
            decoder_layers.append(AELayer(input_dim, input_dim * 2))
        self.decoder_backbone = nn.Sequential(*decoder_layers)
        self.output_proj = nn.Linear(input_dim, input_dim)

    def encode(self, x):
        h = self.encoder_backbone(x)
        return self.latent_proj(h)

    def decode(self, z):
        h = self.latent_expand(z)
        h = self.decoder_backbone(h)
        return self.output_proj(h)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class ActionDecoder(nn.Module):
    """
    Decodes high-bandwidth latent thought vectors into discrete trading actions.
    Maps 128D -> 3-way distribution (Long, Neutral, Short/Hedge).
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # 0: Neutral, 1: Long, 2: Short/Hedge
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class BrierLoss(nn.Module):
    """
    Likelihood-free loss for continuous latent vectors.
    Based on the paper's collision probability concept.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z_pred, z_target):
        # In a regression context, Brier is often equivalent to MSE 
        # but the CALM paper uses it as a collision probability check.
        # For this version, we will implement the MSE as a primary metric 
        # and include a collision check as a secondary term.
        mse_loss = F.mse_loss(z_pred, z_target)
        # Collision term: how similar are they in cosine space?
        cos_sim = F.cosine_similarity(z_pred, z_target).mean()
        return mse_loss + (1.0 - cos_sim)
