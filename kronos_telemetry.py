import sys
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from tqdm import tqdm

# Connect Kronos repo
sys.path.append("/Users/milocobb/tft_model/Kronos_repo")
from model import KronosTokenizer


def main():
    print("Loading Kronos Tokenizer...")
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(
        device
    )
    tokenizer.eval()

    print("Downloading historical SPY data (1998 to 2026)...")
    spy = yf.download("SPY", start="1998-01-01", end="2026-01-01", progress=False)

    # Make sure we don't have multi-index columns, which yfinance sometimes gives in recent versions.
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()

    # Pre-extract values
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = spy[price_cols].copy().dropna()
    df["Amount"] = df["Volume"] * ((df["Open"] + df["Close"]) / 2.0)

    full_vals = df[["Open", "High", "Low", "Close", "Volume", "Amount"]].values.astype(
        np.float32
    )

    # We will build sliding sequence windows
    max_context = 512
    N = len(df)

    s1_tokens = []
    s2_tokens = []
    valid_indices = []

    print(f"Total trading days: {N}")
    print("Encoding historical candlesticks into hierarchical tokens (causal mode)...")

    # We will run this in batches for speed, but each window is strictly [i-max_context : i]
    # To batch effectively, we will construct a batch tensor of shape [batch_size, max_context, 6]

    # For every day t >= 512, we take [t-512:t]
    # For t < 512, we can just take [0:t] (Kronos handles variable lengths or we can pad, but padding might break it).
    # Since Kronos tokenizer accepts different lengths, let's just do it individually for small t, and batched for t >= 512.
    # To keep it simple, we only start emitting tokens when we have at least 120 days.

    for i in tqdm(range(120, N)):
        start_idx = max(0, i - max_context + 1)  # max 512 length up to current day `i`
        end_idx = i + 1

        window = full_vals[start_idx:end_idx]

        # Predictor normalization logically happens here (cross-sectional normalizer)
        w_mean = np.mean(window, axis=0)
        w_std = np.std(window, axis=0)
        w_norm = (window - w_mean) / (w_std + 1e-5)
        w_norm = np.clip(w_norm, -5, 5)

        # Batch size 1, sequence length `L`, feats 6
        x_tensor = torch.tensor(w_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            tokens = tokenizer.encode(x_tensor, half=True)

        # The tokens shape is [1, L]. We want the token generated for the LAST timestep in the window.
        s1_t = tokens[0][0, -1].item()
        s2_t = tokens[1][0, -1].item()

        s1_tokens.append(s1_t)
        s2_tokens.append(s2_t)
        valid_indices.append(df.index[i])

    print("Integration complete. Extracted", len(s1_tokens), "token pairs.")

    # Create DataFrame of tokens
    tokens_df = pd.DataFrame(
        {"Kronos_S1": s1_tokens, "Kronos_S2": s2_tokens}, index=valid_indices
    )

    print("Merging with main parquet...")
    parquet_path = "/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet"
    master_df = pd.read_parquet(parquet_path)
    # the index of master_df is DatetimeIndex.

    # Left join onto master_df
    master_df = master_df.join(tokens_df, how="left")

    # Fill Nans (early history < 120 days)
    master_df["Kronos_S1"] = (
        master_df["Kronos_S1"].fillna(0).astype("int").astype("str")
    )
    master_df["Kronos_S2"] = (
        master_df["Kronos_S2"].fillna(0).astype("int").astype("str")
    )

    master_df.to_parquet(parquet_path)
    print(
        f"Saved {parquet_path} with new string categorical columns: Kronos_S1, Kronos_S2."
    )
    print("Sample:\n", master_df[["SPY", "Kronos_S1", "Kronos_S2"]].tail())


if __name__ == "__main__":
    main()
