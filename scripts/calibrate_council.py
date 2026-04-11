import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yfinance as yf

# Ensure correct pathing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.deliberation import LatentCouncil


def get_calibration_data():
    print("[Calibration] Fetching historical data (2000-2018)...")
    tickers = {"SPY": "SPY", "VIX": "^VIX", "GOLD": "GC=F", "COPPER": "HG=F"}

    dfs = {}
    for name, ticker in tickers.items():
        data = yf.download(ticker, start="2000-01-01", end="2018-12-31")
        dfs[name] = data["Close"]

    df = pd.concat(dfs.values(), axis=1)
    df.columns = tickers.keys()
    df = df.ffill().dropna()

    # 1. Generate Ground Truth Labels (20-day forward returns)
    # Target: 0 (Neutral), 1 (Long), 2 (Short/Hedge)
    df["forward_return"] = df["SPY"].pct_change(20).shift(-20)

    threshold = 0.02  # 2% move
    df["label"] = 0  # Default Neutral
    df.loc[df["forward_return"] > threshold, "label"] = 1  # Long
    df.loc[df["forward_return"] < -threshold, "label"] = 2  # Short

    # Drop rows without labels (end of dataset)
    df = df.dropna()

    print(f"[Calibration] Dataset ready. Total samples: {len(df)}")
    print(f"Distribution: {df['label'].value_counts().to_dict()}")

    return df


def train_council(df, epochs=50):
    feature_dim = 4
    council = LatentCouncil(feature_dim=feature_dim)

    optimizer = optim.Adam(
        list(council.model.parameters())
        + list(council.role_adapter.parameters())
        + list(council.action_decoder.parameters()),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    # Convert to Tensors
    features = torch.tensor(
        df[["SPY", "VIX", "GOLD", "COPPER"]].values, dtype=torch.float32
    )
    labels = torch.tensor(df["label"].values, dtype=torch.long)

    print(f"[Calibration] Starting training for {epochs} epochs...")

    council.model.train()
    council.role_adapter.train()
    council.action_decoder.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. Latent Projection (skip Sequential Protocol for simplicity in base calibration)
        latent = council.model.encode(features)

        # 2. Action Decoding
        logits = council.action_decoder(latent)

        # 3. Loss Calculation
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            # Simple accuracy check
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean().item()
            print(
                f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}"
            )

    print("[Calibration] Training complete.")
    council.save_weights()
    return council


if __name__ == "__main__":
    data = get_calibration_data()
    train_council(data)
