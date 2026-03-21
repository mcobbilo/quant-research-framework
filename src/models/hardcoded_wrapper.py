import numpy as np
import pandas as pd
import yfinance as yf
from scipy import signal
from core.context_injector import DynamicContextInjector

def attach_features(df):
    """
    Vectorized calculation of all required technical indicators 
    across the entire historical dataframe to radically speed up evaluation.
    """
    df['SMA_200'] = df['SPY'].rolling(200).mean()
    
    # RSI 5
    delta = df['SPY'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    rs = gain / loss
    df['RSI_5'] = 100 - (100 / (1 + rs))
    
    # Strategy C: Bollinger Bands
    df['SMA_20'] = df['SPY'].rolling(20).mean()
    df['STD_20'] = df['SPY'].rolling(20).std()
    df['Lower_BB'] = df['SMA_20'] - (2 * df['STD_20'])
    
    # Strategy B: Volatility Exhaustion
    df['VIX_prev'] = df['VIX'].shift(1)
    df['SPY_ret_prev_1'] = df['SPY'].pct_change().shift(1)
    df['SPY_ret_prev_2'] = df['SPY'].pct_change().shift(2)
    df['SPY_ret_prev_3'] = df['SPY'].pct_change().shift(3)
    
    # Phase 14: Dr. Copper vs Gold Macro Filter
    df['COPPER_GOLD_RATIO'] = df['COPPER'] / df['GOLD']
    df['CGR_SMA_200'] = df['COPPER_GOLD_RATIO'].rolling(200).mean()
    
    # Strategy D: Fourier Aftershock
    df['Z_Score'] = (df['SPY'] - df['SMA_20']) / df['STD_20']
    
    # Phase 15: VIX Technical Engineering
    # VIX 7-Day Custom Percentage Price Oscillator (Fast=1, Slow=7)
    ema_fast = df['VIX'].ewm(span=1, adjust=False).mean()
    ema_slow = df['VIX'].ewm(span=7, adjust=False).mean()
    df['VIX_PPO_7'] = ((ema_fast - ema_slow) / ema_slow) * 100
    
    # Phase 20: Attach Proprietary SP500 McClellan Oscillator Proxy
    try:
        mco_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'mcclellan_sp500.csv')
        mco_df = pd.read_csv(mco_path, index_col=0, parse_dates=True)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        if mco_df.index.tz is not None: mco_df.index = mco_df.index.tz_localize(None)
        df['MCO'] = mco_df['MCO']
        df.ffill(inplace=True)
    except Exception:
        df['MCO'] = 0.0
    
    # VIX Bollinger Bands (20-day moving average, 2 Standard Deviations)
    df['VIX_SMA_20'] = df['VIX'].rolling(20).mean()
    df['VIX_STD_20'] = df['VIX'].rolling(20).std()
    df['VIX_BB_LOWER'] = df['VIX_SMA_20'] - (2 * df['VIX_STD_20'])
    df['VIX_BB_UPPER'] = df['VIX_SMA_20'] + (2 * df['VIX_STD_20'])
    
    return df

class StrategyA:
    def __init__(self):
        self.name = "Strategy A: Bull Market Pullback"
        
    def evaluate(self, row):
        # High VIX, Bull Market Trend, Severe short-term oversold condition
        c1 = row['VIX'] > 25
        c2 = row['SPY'] > row['SMA_200']
        c3 = row['RSI_5'] < 30
        return 1.0 if (c1 and c2 and c3) else 0.50

class StrategyB:
    def __init__(self):
        self.name = "Strategy B: Volatility Exhaustion"
        
    def evaluate(self, row):
        # Massive panic, but falling VIX + 3 days of localized price dumping
        c1 = row['VIX'] > 35
        c2 = row['VIX'] < row['VIX_prev']
        c3 = (row['SPY_ret_prev_1'] < 0) and (row['SPY_ret_prev_2'] < 0) and (row['SPY_ret_prev_3'] < 0)
        return 1.0 if (c1 and c2 and c3) else 0.50

class StrategyC:
    def __init__(self):
        self.name = "Strategy C: Structurally Broken Vol Expansion"
        
    def evaluate(self, row):
        # Fear + Severe deviation below 2 standard deviations
        c1 = row['VIX'] > 20
        c2 = row['SPY'] < row['Lower_BB']
        return 1.0 if (c1 and c2) else 0.50

class StrategyD:
    def __init__(self, entry_z=-3.0, exit_z=2.5, baseline_prob=0.75):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.baseline_prob = baseline_prob
        self.name = f"Phase 9: SPY Baseline + {abs(entry_z)} Sigma Aftershock"
        self.in_trade = False
        self.injector = DynamicContextInjector()
        
    def evaluate(self, row):
        z = row['Z_Score']
        
        # If we are currently in our resting baseline (1.0x SPY)
        if not self.in_trade:
            if z < self.entry_z: 
                self.in_trade = True
                
                # Fetch date for the dynamic context injection log
                if hasattr(row, 'name'):
                    date_str = row.name.strftime('%Y-%m-%d')
                else:
                    date_str = "Unknown Date"
                    
                # Harness Architecture: Radically narrow the LLM scope exactly at the point of failure
                self.injector.query_macro_layer(date_str, z)
                
                return 1.0 # Max leverage (2.0x)
            return self.baseline_prob
        
        # If we are currently holding the 2.0x leveraged position
        else:
            if z > self.exit_z: 
                self.in_trade = False
                return self.baseline_prob
            return 1.0 # Continue holding through the chop

class StrategyE:
    def __init__(self, entry_z=-3.5, hold_days=60, baseline_prob=0.75):
        self.entry_z = entry_z
        self.hold_days = hold_days
        self.baseline_prob = baseline_prob
        self.name = f"Strategy E (Wavelet {hold_days}-Day Decay)"
        self.days_in_trade = 0
        self.in_trade = False
        
    def evaluate(self, row):
        z = row['Z_Score']
        
        # Resting Baseline (1.0x SPY)
        if not self.in_trade:
            if z < self.entry_z:
                self.in_trade = True
                self.days_in_trade = 1
                return 1.0 # Fire 2.0x Kelly leverage
            return self.baseline_prob
            
        # Time-bound Hold
        else:
            self.days_in_trade += 1
            if self.days_in_trade >= self.hold_days:
                # The 60-day Wavelet ripple has mathematically decayed. Exit leverage.
                self.in_trade = False
                self.days_in_trade = 0
                return self.baseline_prob
            return 1.0 # Hold the leverage through the harmonic aftershock

class StrategyF:
    """
    Damped Harmonic Oscillator
    Models the 60-day volatility decay as an exponentially shrinking envelope.
    Actively scalps the secondary ripples instead of statically holding.
    """
    def __init__(self, shock_z=-3.5, decay_days=60, baseline_prob=0.75):
        self.shock_z = shock_z # Magnitude of initial crash trigger
        self.decay_days = decay_days # Duration of the wavelet aftershock timeline
        self.baseline_prob = baseline_prob
        self.name = f"Strategy F (Harmonic Decay Scalper)"
        
        self.active_cycle = False
        self.t = 0 # Days since crash (time variable for decay function)
        self.in_trade = False
        
    def evaluate(self, row):
        z = row['Z_Score']
        import math
        
        # If the market is perfectly stable and we aren't tracking a crash cycle
        if not self.active_cycle:
            if z < self.shock_z:
                # Boom. Earthquake trigger.
                self.active_cycle = True
                self.t = 1
                self.in_trade = True # Buy the absolute bottom
                return 1.0 # 2.0x Margin Leverage
            return self.baseline_prob
            
        else:
            # We are actively inside the 60-day harmonic decay window
            self.t += 1
            
            # If 60-days pass, the cycle formally resets
            if self.t >= self.decay_days:
                self.active_cycle = False
                self.in_trade = False
                self.t = 0
                return self.baseline_prob
                
            # Calculate the dynamically decaying bounds for today 't'
            # Formula: Peak_Amplitude * e^(-lambda * t)
            # We decay from an upper bound of +3.0 down to +0.2 over 60 days
            # We decay from a lower bound of -3.5 down to -0.2 over 60 days
            lambda_factor = 2.5 / self.decay_days # Calibrated exponential decay rate
            
            upper_bound = 3.0 * math.exp(-lambda_factor * self.t)
            lower_bound = -3.5 * math.exp(-lambda_factor * self.t)
            
            if self.in_trade:
                # We currently hold 2.0x Kelly leverage. Looking to scalp the bounce (sell)
                if z > upper_bound:
                    self.in_trade = False
                    return self.baseline_prob # Sell the bounce, revert to 1.0x SPY
                return 1.0 # Hold during the localized drop
            else:
                # We sold the previous bounce, waiting for the secondary 
                # (but shallower) downward aftershock to re-enter!
                if z < lower_bound:
                    self.in_trade = True
                    return 1.0 # Re-buy the secondary crash
                return self.baseline_prob

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
        z = row['Z_Score']
        cgr = row['COPPER_GOLD_RATIO']
        cgr_sma = row['CGR_SMA_200']
        
        # Resting Baseline (1.0x SPY)
        if not self.in_trade:
            if z < self.entry_z: 
                # Macro Block: We only double down margin IF Dr. Copper implies industrial growth!
                if cgr > cgr_sma:
                    self.in_trade = True
                    
                    if hasattr(row, 'name'):
                        date_str = row.name.strftime('%Y-%m-%d')
                    else:
                        date_str = "Unknown Date"
                        
                    self.injector.query_macro_layer(date_str, z)
                    return 1.0 # Max leverage (2.0x)
                else:
                    # RECESSION RISK: Dr. Copper says the crash is fundamentally real. Ignore the bounce.
                    return self.baseline_prob
                    
            return self.baseline_prob
        
        # If we are currently holding the 2.0x leveraged position
        else:
            if z > self.exit_z: 
                self.in_trade = False
                return self.baseline_prob
            return 1.0 # Continue holding through the chop

