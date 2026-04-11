import numpy as np
import yfinance as yf
import pywt


def analyze_wavelet():
    print(
        "[Wavelet Engine] Initiating Morlet Wavelet Transform Analysis on VIX Aftershocks..."
    )

    vix = yf.download("^VIX", start="2019-01-01", end="2021-01-01", progress=False)
    if hasattr(vix.columns, "levels"):
        vix_vals = vix[("Close", "^VIX")].dropna().values
    else:
        vix_vals = vix["Close"].dropna().values

    print(
        "[Wavelet Engine] Data loaded: 2019-2021 (Targeting the massive 2020 COVID Crash)."
    )

    widths = np.arange(1, 61)

    # pywt.cwt computes continuous wavelet transform
    cwtmatr, freqs = pywt.cwt(vix_vals - np.mean(vix_vals), widths, "morl")

    crash_idx = np.argmax(vix_vals)
    print(
        f"\n[Matrix] VIX Crash Peak Detected at Day Index {crash_idx} (Absolute VIX: {vix_vals[crash_idx]:.2f})"
    )

    aftershock_window = 30
    if crash_idx + aftershock_window < len(vix_vals):
        aftershock_cwt = np.abs(cwtmatr[:, crash_idx : crash_idx + aftershock_window])
        mean_power = np.mean(aftershock_cwt, axis=1)
        dominant_width = widths[np.argmax(mean_power)]

        print("\n================= WAVELET AFTERSHOCK RESULTS =================")
        print(
            f"During the extreme 30-day Post-Crash aftermath, the VIX oscillated with a dominant, repeating ripple period of exactly: {dominant_width} Trading Days."
        )

        quiet_idx = 150  # roughly mid 2019
        quiet_cwt = np.abs(cwtmatr[:, quiet_idx : quiet_idx + aftershock_window])
        quiet_width = widths[np.argmax(np.mean(quiet_cwt, axis=1))]

        print(
            f"For comparison, during the quiet, stable Bull Market of 2019, the dominant Volatility ripple period collapsed down to a baseline of: {quiet_width} Trading Days."
        )
        print("============================================================\n")


if __name__ == "__main__":
    analyze_wavelet()
