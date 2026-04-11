import pandas as pd
from core.context_injector import DynamicContextInjector


def attach_features(df):
    """
    Vectorized calculation of all required technical indicators
    across the entire historical dataframe to radically speed up evaluation.
    """
    df["SMA_200"] = df["SPY"].rolling(200).mean()

    # RSI 5
    delta = df["SPY"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    rs = gain / loss
    df["RSI_5"] = 100 - (100 / (1 + rs))

    # Strategy C: Bollinger Bands
    df["SMA_20"] = df["SPY"].rolling(20).mean()
    df["STD_20"] = df["SPY"].rolling(20).std()
    df["Lower_BB"] = df["SMA_20"] - (2 * df["STD_20"])

    # Strategy B: Volatility Exhaustion
    df["VIX_prev"] = df["VIX"].shift(1)
    df["SPY_ret_prev_1"] = df["SPY"].pct_change().shift(1)
    df["SPY_ret_prev_2"] = df["SPY"].pct_change().shift(2)
    df["SPY_ret_prev_3"] = df["SPY"].pct_change().shift(3)

    # Phase 14: Dr. Copper vs Gold Macro Filter
    df["COPPER_GOLD_RATIO"] = df["COPPER"] / df["GOLD"]
    df["CGR_SMA_200"] = df["COPPER_GOLD_RATIO"].rolling(200).mean()

    # Strategy D: Fourier Aftershock
    df["Z_Score"] = (df["SPY"] - df["SMA_20"]) / df["STD_20"]

    # Phase 15: VIX Technical Engineering
    # VIX 7-Day Custom Percentage Price Oscillator (Fast=1, Slow=7)
    ema_fast = df["VIX"].ewm(span=1, adjust=False).mean()
    ema_slow = df["VIX"].ewm(span=7, adjust=False).mean()
    df["VIX_PPO_7"] = ((ema_fast - ema_slow) / ema_slow) * 100

    # Phase 20: Attach Proprietary SP500 McClellan Oscillator Proxy
    try:
        mco_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "mcclellan_sp500.csv",
        )
        mco_df = pd.read_csv(mco_path, index_col=0, parse_dates=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if mco_df.index.tz is not None:
            mco_df.index = mco_df.index.tz_localize(None)
        df["MCO"] = mco_df["MCO"]
        df.ffill(inplace=True)
    except Exception:
        df["MCO"] = 0.0

    # VIX Bollinger Bands (20-day moving average, 2 Standard Deviations)
    df["VIX_SMA_20"] = df["VIX"].rolling(20).mean()
    df["VIX_STD_20"] = df["VIX"].rolling(20).std()
    df["VIX_BB_LOWER"] = df["VIX_SMA_20"] - (2 * df["VIX_STD_20"])
    df["VIX_BB_UPPER"] = df["VIX_SMA_20"] + (2 * df["VIX_STD_20"])

    # Phase 24: Signal Lagging for Roan Combiner
    df["SMA_200_prev"] = df["SMA_200"].shift(1)
    df["SMA_20_prev"] = df["SMA_20"].shift(1)
    df["GOLD_SMA_200"] = df["GOLD"].rolling(200).mean()
    df["COPPER_SMA_200"] = df["COPPER"].rolling(200).mean()
    df["DAILY_RET"] = df["SPY"].pct_change()

    return df


class StrategyD:
    def __init__(self, entry_z=-3.0, exit_z=2.5, baseline_prob=0.75):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.baseline_prob = baseline_prob
        self.name = f"Phase 9: SPY Baseline + {abs(entry_z)} Sigma Aftershock"
        self.in_trade = False
        self.injector = DynamicContextInjector()

    def evaluate(self, row):
        z = row["Z_Score"]

        # If we are currently in our resting baseline (1.0x SPY)
        if not self.in_trade:
            if z < self.entry_z:
                self.in_trade = True

                # Fetch date for the dynamic context injection log
                if hasattr(row, "name"):
                    date_str = row.name.strftime("%Y-%m-%d")
                else:
                    date_str = "Unknown Date"

                # Harness Architecture: Radically narrow the LLM scope exactly at the point of failure
                self.injector.query_macro_layer(date_str, z)

                return 1.0  # Max leverage (2.0x)
            return self.baseline_prob

        # If we are currently holding the 2.0x leveraged position
        else:
            if z > self.exit_z:
                self.in_trade = False
                return self.baseline_prob
            return 1.0  # Continue holding through the chop


class StrategyG:
    """
    Phase 14: Dr. Copper Global Manufacturing Filter
    Identical strictly to Strategy D (Phase 9 Yield), but mathematically
    rejects executing margin deployment if the global industrial economy
    is in a systemic downtrend relative to gold (flight to safety).
    """

    def __init__(self, entry_z=-3.5, exit_z=3.0, baseline_prob=0.75):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.baseline_prob = baseline_prob
        self.name = f"Phase 14: Dr. Copper Ratio ({abs(entry_z)} Sigma Filter)"
        self.in_trade = False
        self.injector = DynamicContextInjector()

    def evaluate(self, row):
        z = row["Z_Score"]
        cgr = row["COPPER_GOLD_RATIO"]
        cgr_sma = row["CGR_SMA_200"]

        # Resting Baseline (1.0x SPY)
        if not self.in_trade:
            if z < self.entry_z:
                # Macro Block: We only double down margin IF Dr. Copper implies industrial growth!
                if cgr > cgr_sma:
                    self.in_trade = True

                    if hasattr(row, "name"):
                        date_str = row.name.strftime("%Y-%m-%d")
                    else:
                        date_str = "Unknown Date"

                    self.injector.query_macro_layer(date_str, z)
                    return 1.0  # Max leverage (2.0x)
                else:
                    # RECESSION RISK: Dr. Copper says the crash is fundamentally real. Ignore the bounce.
                    return self.baseline_prob

            return self.baseline_prob

        # If we are currently holding the 2.0x leveraged position
        else:
            if z > self.exit_z:
                self.in_trade = False
                return self.baseline_prob
            return 1.0  # Continue holding through the chop


class MetaStrategyClassifier:
    """
    Phase 97: Master Regime Classifier (Encyclopedia Matrix)
    Determines absolute execution state based on VIX Term Structure and SMA Expansion.
    """

    def __init__(self):
        # We explicitly mandate a 0.49 (Cash) resting state. During Panic or Chop,
        # we pull out of the SPY index and sit fully in 0.0x cash unless an extrema buy-dip signal is hit.
        self.strategy_mean_reversion = StrategyD(
            baseline_prob=0.49
        )  # Phase 9 SPY Extremum Dip
        self.strategy_stat_arb = StrategyG(
            baseline_prob=0.49
        )  # Phase 14 Dr Copper Pairs Filter

    def evaluate_regime(self, row):
        # 1. PANIC (VIX Backwardation OR Structural Crash)
        if (
            row.get("VIX_TERM_STRUCTURE_6M", 0) > 1.0
            or row.get("SPY_PCT_FROM_200", 0) < -5.0
        ):
            return "MEAN_REVERSION_PANIC"
        # 2. EXPANSION (VIX Contango AND Structural Expansion)
        elif (
            row.get("VIX_TERM_STRUCTURE_6M", 1) < 0.90
            and row.get("SPY_PCT_FROM_200", 0) > 1.0
        ):
            return "TREND_FOLLOWING_EXPANSION"
        # 3. SIDEWAYS (Baseline Float)
        else:
            return "STAT_ARB_CHOP"

    def evaluate(self, row):
        regime = self.evaluate_regime(row)

        if regime == "MEAN_REVERSION_PANIC":
            return self.strategy_mean_reversion.evaluate(row)
        elif regime == "STAT_ARB_CHOP":
            return self.strategy_stat_arb.evaluate(row)
        else:
            # TREND_FOLLOWING_EXPANSION: Hold SPY at maximum baseline safety without exotic leverage.
            return 1.0
