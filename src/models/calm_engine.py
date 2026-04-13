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
    Stochastic Variational AutoEncoder for high-bandwidth market state representation.
    Compresses high-dimensional technical and OSINT features into a latent probabilistic space.
    Adapted from CALM (arXiv:2510.27688).
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
        self.latent_proj = nn.Linear(input_dim, latent_dim * 2) # Expand for mean and log_std

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
        latent_states = self.encode(x)
        mean, log_std = torch.chunk(latent_states, 2, dim=-1)
        
        # Reparameterization
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        z = mean + eps * std if self.training else mean
        
        # Precompute KL Divergence penalty
        kl_loss = 0.5 * (torch.pow(mean, 2) + torch.pow(std, 2) - 1 - log_std * 2)
        kl_loss = torch.clamp(kl_loss, min=-20.0)
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))
        
        x_recon = self.decode(z)
        return x_recon, mean, log_std, kl_loss


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
            nn.Linear(32, 3),  # 0: Neutral, 1: Long, 2: Short/Hedge
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class EnergyLoss(nn.Module):
    """
    Likelihood-free Energy Score for continuous latent generation.
    Extracted from the official CALM paper implementation (shaochenze/calm).
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def distance(self, x_1, x_2):
        return torch.pow(torch.linalg.norm(x_1 - x_2, ord=2, dim=-1), self.beta)
    
    def forward(self, x, mean, log_std):
        if x.dim() == mean.dim():
            x = x.unsqueeze(dim=0)  # Add n_x dimension for multiple prediction paths
            
        n_x = x.shape[0]
        x_i = x.unsqueeze(1)  # (n_x, 1, batch_size, ...)
        x_j = x.unsqueeze(0)  # (1, n_x, batch_size, ...)
        distance_matrix = self.distance(x_i, x_j)
        
        # Penalize collapse across multiple sampled predictions (if n_x > 1)
        if n_x > 1:
            distance_x = distance_matrix.sum(dim=(0,1)) / (n_x * (n_x - 1))
        else:
            distance_x = 0.0

        std = torch.exp(log_std)
        n_y = 100 # Emulate density 
        eps = torch.randn((n_y, *mean.shape), device=mean.device)
        y = mean + eps * std  # (n_y, batch_size, ...)

        x_ = x.reshape(n_x, 1, *x.shape[1:])  # (n_x, 1, batch_size, ...)
        y_ = y.reshape(1, n_y, *y.shape[1:])  # (1, n_y, batch_size, ...)
        
        # Minimize distance to generated synthetic ground-truth space
        distance_y = self.distance(x_, y_).mean(dim=(0, 1))
        
        score = distance_x - distance_y * 2
        return score
