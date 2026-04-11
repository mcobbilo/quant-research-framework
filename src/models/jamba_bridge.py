import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
import math

# --- Architecture Definition (Localized for Bridge Stability) ---


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        return self.w3(gate * self.w2(x))


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = SwiGLU(d_model, int(d_model * 4 * 2 / 3))
        self.ln_1 = RMSNorm(d_model)
        self.ln_2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.size()
        x_norm = self.ln_1(x)
        qkv = self.c_attn(x_norm)
        q, k, v = qkv.split(self.d_model, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(
            1, 1, T, T
        )
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return x + self.dropout(self.c_proj(y)) + self.dropout(self.mlp(self.ln_2(x)))


class MambaBlockProxy(nn.Module):
    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.ln_1 = RMSNorm(d_model)
        self.ln_2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, int(d_model * 4 * 2 / 3))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.size()
        xz = self.in_proj(self.ln_1(x))
        x_p, z = xz.split(self.d_inner, dim=-1)
        x_conv = self.conv1d(x_p.transpose(1, 2))[:, :, :T].transpose(1, 2)
        out = self.out_proj(F.silu(x_conv) * F.silu(z))
        return x + self.dropout(out) + self.dropout(self.mlp(self.ln_2(x)))


class HybridJamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        attn_layer_offset: int = 7,
        num_classes: int = 5,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if (i + 1) % (attn_layer_offset + 1) == 0:
                self.blocks.append(AttentionBlock(d_model, 8))
            else:
                self.blocks.append(MambaBlockProxy(d_model))
        self.lm_head = nn.Linear(d_model, num_classes, bias=False)
        self.final_norm = RMSNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


# --- Jamba Bridge Logic ---


class JambaBridge:
    def __init__(self, checkpoint_frame=2252):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = self._init_model(checkpoint_frame)
        self.macro_proj = nn.Conv1d(16, 16, 1).to(self.device)
        self.options_proj = nn.Conv1d(48, 16, 1).to(self.device)
        self._load_weights(checkpoint_frame)

    def _init_model(self, frame):
        # 4 layers matching hst_framework/src/execution/eval_gpu_alpha.py:44
        return (
            HybridJamba(d_model=32, num_layers=4, num_classes=5).to(self.device).eval()
        )

    def _load_weights(self, frame):
        # Weights located in hst_framework relative to project sibling
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        base_dir = os.path.dirname(root_dir)
        ckpt_path = os.path.join(
            base_dir,
            "hst_framework",
            "models",
            "checkpoints",
            f"hst_alpha_cluster_frame_{frame}.pt",
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"HST Alpha weights not found at: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.macro_proj.load_state_dict(checkpoint["macro_proj"])
        self.options_proj.load_state_dict(checkpoint["options_proj"])
        self.model.load_state_dict(checkpoint["jamba"])
        print(
            f"[JambaBridge] Successfully loaded weights from Frame {frame} on {self.device.type.upper()}."
        )

    def _get_live_tape(self, lookback=16):
        # Increased period to 2y to support SMA-200 calculation
        spy_raw = yf.download("SPY", period="2y", progress=False, auto_adjust=True)
        vix_raw = yf.download("^VIX", period="2y", progress=False)

        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy = spy_raw["Close"]["SPY"]
            vix = vix_raw["Close"]["^VIX"]
        else:
            spy = spy_raw["Close"]
            vix = vix_raw["Close"]

        df = pd.DataFrame({"SPY": spy, "VIX": vix}).ffill().dropna()
        df["Returns"] = df["SPY"].pct_change()
        df["VIX_Change"] = df["VIX"].pct_change()

        features = ["Returns", "VIX", "VIX_Change"]
        for w in [5, 15, 30, 60, 200]:
            df[f"SMA_{w}"] = df["SPY"].rolling(w).mean() / df["SPY"] - 1.0
            features.append(f"SMA_{w}")
        for w in [5, 15, 30, 60]:
            df[f"VOL_{w}"] = df["Returns"].rolling(w).std()
            features.append(f"VOL_{w}")
        # RSI
        delta = df["SPY"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI_14"] = 100 - (100 / (1 + (gain / loss))).fillna(50) / 100.0
        features.append("RSI_14")

        # EXACTLY 32 features required by HST weights
        for lag in range(1, 50):  # Increased range to guarantee 32
            if len(features) >= 32:
                break
            df[f"LAG_{lag}"] = df["Returns"].shift(lag)
            features.append(f"LAG_{lag}")

        df = df.dropna().tail(lookback)
        feature_matrix = df[features].values  # [16, 32]
        return torch.tensor(feature_matrix.T, dtype=torch.float32).unsqueeze(0).to(
            self.device
        ), df["VIX"].iloc[-1]

    def _construct_options_tensor(self, last_vix, lookback=16):
        implied_daily = last_vix / np.sqrt(252) / 100.0
        X_o = np.zeros((1, lookback, 12, 4))
        for t in range(lookback):
            for node in range(12):
                strike_dist = ((node - 5.5) / 5.5) * implied_daily * 5.0
                X_o[0, t, node, 0] = strike_dist
                X_o[0, t, node, 1] = last_vix + strike_dist * 0.1
                X_o[0, t, node, 2] = 1.0 / (1.0 + np.exp(-strike_dist * 50))
                X_o[0, t, node, 3] = np.exp(-0.5 * (strike_dist * 100) ** 2)
        return torch.tensor(X_o, dtype=torch.float32).to(self.device)

    def get_latest_alpha_signal(self):
        with torch.no_grad():
            X_m, last_vix = self._get_live_tape()
            X_o = self._construct_options_tensor(last_vix)

            # Forward Pass Logic from eval_gpu_alpha
            macro_emb = (
                self.macro_proj(X_m.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
            )
            b, t, n, f = X_o.shape
            options_flat = X_o.view(b, t, -1)
            option_emb = (
                self.options_proj(options_flat.permute(0, 2, 1))
                .permute(0, 2, 1)
                .contiguous()
            )

            # Repeat to align time-dimensions
            if option_emb.size(1) < macro_emb.size(1):
                option_emb = option_emb.repeat_interleave(
                    macro_emb.size(1) // option_emb.size(1), dim=1
                )

            fused = torch.cat([macro_emb, option_emb], dim=-1)
            logits = self.model(fused)[:, -1, :]  # Last entry
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            regimes = [
                "Massive Left-Tail",
                "Negative Bias",
                "Neutral/Flat",
                "Positive Bias",
                "Massive Right-Tail",
            ]
            pred_idx = np.argmax(probs)

            return {
                "ensemble_prediction": regimes[pred_idx],
                "confidence": float(probs[pred_idx]),
                "distribution": {r: float(p) for r, p in zip(regimes, probs)},
            }


if __name__ == "__main__":
    bridge = JambaBridge()
    signal = bridge.get_latest_alpha_signal()
    print(f"Signal Result: {signal}")
