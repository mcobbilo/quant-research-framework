import os
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from lightgbm import LGBMClassifier
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings, logging, joblib

warnings.filterwarnings('ignore')

# ========================== CONFIG ==========================
ASSETS = ['SPY', 'TLT', 'GLD']
START_DATE = '2016-01-01'
FORECAST_HORIZON = 5
REBALANCE_DAYS = 5
TRAIN_WINDOW = 1000
SWITCH_THRESHOLD = 0.002
TC = 0.0005                   # 5 bp one-way
TARGET_VOL = 0.13             # 13% annualized target
VOL_LOOKBACK = 21
MAX_EXPOSURE = 1.0
RANDOM_STATE = 42
MODEL_DIR = "saved_models/"
NUM_SEATS = 50                # The Senate

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"🚀 FINAL v5.0 THE SENATE (Target Vol: {TARGET_VOL*100}%)")

# ====================== 1. DATA ======================
extra_assets = ['DX-Y.NYB', 'EEM', 'USO', 'CPER', '^MOVE', '^VIX3M', 'HYG', 'HG=F', '^VVIX', '^SKEW', 'IWM']
download_tickers = ['SPY', 'TLT', 'GLD'] + extra_assets
prices = yf.download(download_tickers, start=START_DATE)['Close'].asfreq('B').ffill().astype(np.float32)
prices.rename(columns={'DX-Y.NYB': 'DXY', '^MOVE': 'MOVE', '^VIX3M': 'VIX3M', 'HG=F': 'HGF', '^VVIX': 'VVIX', '^SKEW': 'SKEW'}, inplace=True)
returns = np.log(prices / prices.shift(1)).fillna(0).astype(np.float32)

def get_fred_data(tickers, start_date):
    import requests, io
    dfs = []
    for ticker in tickers:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"
        df = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')
        df = df[df.index >= start_date]
        dfs.append(df)
    return pd.concat(dfs, axis=1)

macro = get_fred_data(['VIXCLS', 'T10Y2Y', 'DTB3', 'BAMLH0A0HYM2', 'T10YIE', 'DGS10', 'T10Y3M', 'BAA10Y', 'NFCI', 'TEDRATE'], START_DATE)
macro.columns = ['vix', 'yield_curve', 'rf_rate', 'hy_spread', 'inflation_exp', 'dgs10', 't10y3m', 'baa10y', 'nfci', 'ted_spread']
macro = macro.asfreq('B').ffill()
macro['nfci'] = macro['nfci'].shift(5)  # Fix: Lag NFCI by 5 business days
macro = macro.reindex(prices.index, method='ffill').bfill()

rf_daily = (macro['rf_rate'] / 100) / 252

# ====================== 2. SOTA FEATURES ======================
def create_features(prices, returns, macro):
    df = pd.DataFrame(index=prices.index)
    df = pd.concat([df, returns.add_suffix('_ret')], axis=1)
    
    mom5 = returns.rolling(5).sum().add_suffix('_mom5')
    df = pd.concat([df, mom5], axis=1)
    
    def zscore(s, w=252):
        return (s - s.rolling(w).mean()) / s.rolling(w).std()
    
    for c in mom5.columns:
        df[c + '_z'] = zscore(mom5[c])
        
    df['spy_tlt_ratio_z'] = zscore(prices['SPY'] / prices['TLT'])
    df['spy_gld_ratio_z'] = zscore(prices['SPY'] / prices['GLD'])
    df['tlt_gld_ratio_z'] = zscore(prices['TLT'] / prices['GLD'])
    df['dxy_spy_ratio_z'] = zscore(prices['DXY'] / prices['SPY'])
    
    df['eem_spy_ratio_z'] = zscore(prices['EEM'] / prices['SPY'])
    df['cper_gld_ratio_z'] = zscore(prices['CPER'] / prices['GLD'])
    df['hgf_gld_ratio_z'] = zscore(prices['HGF'] / prices['GLD'])
    df['spy_uso_ratio_z'] = zscore(prices['SPY'] / prices['USO'])
    df['hyg_tlt_ratio_z'] = zscore(prices['HYG'] / prices['TLT'])
    df['spy_hyg_ratio_z'] = zscore(prices['SPY'] / prices['HYG'])
    df['vix_vxv_ratio_z'] = zscore(macro['vix'] / prices['VIX3M'])
    
    df['iwm_spy_ratio_z'] = zscore(prices['IWM'] / prices['SPY'])
    df['vix_vvix_ratio_z'] = zscore(macro['vix'] / prices['VVIX'])
    df['skew_z'] = zscore(prices['SKEW'])
    
    for h in [10, 21, 63, 126, 252]:
        mom = returns.rolling(h).sum().add_suffix(f'_mom{h}')
        df = pd.concat([df, mom], axis=1)
        for c in mom.columns:
            df[c + '_z'] = zscore(mom[c])
    
    accel = (returns.rolling(21).sum() - returns.rolling(5).sum()).add_suffix('_accel')
    df = pd.concat([df, accel], axis=1)
    
    for w in [21, 63, 252]:
        vol = returns.rolling(w).std().add_suffix(f'_vol{w}')
        df = pd.concat([df, vol], axis=1)
        for c in vol.columns:
            df[c + '_z'] = zscore(vol[c])
    
    vol_ratio_df = (returns.rolling(21).std() / returns.rolling(252).std()).add_suffix('_vol_ratio')
    df = pd.concat([df, vol_ratio_df], axis=1)
    
    df['corr_spy_tlt_63'] = returns['SPY'].rolling(63).corr(returns['TLT'])
    df['corr_spy_gld_63'] = returns['SPY'].rolling(63).corr(returns['GLD'])
    df['corr_tlt_gld_63'] = returns['TLT'].rolling(63).corr(returns['GLD'])
    
    df['corr_spy_tlt_63_z'] = zscore(df['corr_spy_tlt_63'])
    df['corr_spy_gld_63_z'] = zscore(df['corr_spy_gld_63'])
    df['corr_tlt_gld_63_z'] = zscore(df['corr_tlt_gld_63'])
    
    for asset in ASSETS:
        a = asset.lower()
        close = prices[asset]
        df[f'{a}_rsi14'] = ta.rsi(close, length=14)
        macd = ta.macd(close)
        if macd is not None and not macd.empty:
            macd_col = [c for c in macd.columns if c.startswith('MACD_')][0]
            df[f'{a}_macd'] = macd[macd_col]
        bb = ta.bbands(close)
        if bb is not None and not bb.empty:
            bb_col = [c for c in bb.columns if c.startswith('BBP_')][0]
            df[f'{a}_bb_percent'] = bb[bb_col]
    
    df = df.join(macro[['vix', 'yield_curve', 'hy_spread', 'inflation_exp', 'dgs10', 't10y3m', 'baa10y', 'nfci', 'ted_spread']])
    df['nfci_z'] = zscore(df['nfci'])
    df['ted_spread_z'] = zscore(df['ted_spread'])
    
    df['real_yield_10y'] = df['dgs10'] - df['inflation_exp']
    df['real_yield_10y_z'] = zscore(df['real_yield_10y'])
    
    df['vix_to_10y'] = df['vix'] / df['dgs10']
    df['vix_to_10y_mom21'] = df['vix_to_10y'] - df['vix_to_10y'].rolling(21).mean()
    
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    df['sin_year'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_year'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['sentiment_proxy'] = zscore(-macro['vix'])
    
    return df.ffill()

features = create_features(prices, returns, macro).astype(np.float32)
feature_cols = list(features.columns)
print(f"✅ Features ready: {len(feature_cols)} columns")

import shap

targets = returns.rolling(FORECAST_HORIZON).sum().shift(-FORECAST_HORIZON).add_suffix('_target5d').astype(np.float32)
data = pd.concat([features, targets], axis=1)

print("Running SHAP feature elimination on historical data...")
shap_train = data.dropna(subset=feature_cols + ['SPY_target5d', 'TLT_target5d', 'GLD_target5d']).head(TRAIN_WINDOW)

shap_xgb_spy = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
shap_xgb_spy.fit(shap_train[feature_cols], shap_train['SPY_target5d'])
shap_values_spy = shap.TreeExplainer(shap_xgb_spy).shap_values(shap_train[feature_cols])

shap_xgb_tlt = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
shap_xgb_tlt.fit(shap_train[feature_cols], shap_train['TLT_target5d'])
shap_values_tlt = shap.TreeExplainer(shap_xgb_tlt).shap_values(shap_train[feature_cols])

shap_xgb_gld = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
shap_xgb_gld.fit(shap_train[feature_cols], shap_train['GLD_target5d'])
shap_values_gld = shap.TreeExplainer(shap_xgb_gld).shap_values(shap_train[feature_cols])

shap_abs_mean = (np.abs(shap_values_spy).mean(axis=0) + np.abs(shap_values_tlt).mean(axis=0) + np.abs(shap_values_gld).mean(axis=0)) / 3.0
shap_df = pd.DataFrame({'feature': feature_cols, 'shap': shap_abs_mean}).sort_values('shap', ascending=False)
feature_cols = shap_df['feature'].head(150).tolist()
print(f"✅ Reduced to Top 150 orthogonal features via SHAP.")

tdfs = []
for h in [1, 2, 3, 5, 10, 21]:
    if h == 5: continue
    tdfs.append(returns.rolling(h).sum().shift(-h).add_suffix(f'_target{h}d').astype(np.float32))
if tdfs:
    data = pd.concat([data] + tdfs, axis=1)

print("Running volatility-targeted zero-leakage SENATE backtest (150 Weak Learners)...")
positions, weights, strategy_period_rets, dates = [], [], [], []

rebalance_idx = list(range(TRAIN_WINDOW, len(data) - FORECAST_HORIZON, REBALANCE_DAYS))
start_exec_date = returns.index[rebalance_idx[0] + 1]

# Set up the Senate Sub-Models (v5.2 Micro-Fractal Final)
np.random.seed(RANDOM_STATE)
shuffled_features = np.random.choice(feature_cols, size=len(feature_cols), replace=False).tolist()
chunk_size = max(1, len(shuffled_features) // NUM_SEATS)
seat_features = [shuffled_features[i*chunk_size : (i+1)*chunk_size] for i in range(NUM_SEATS)]
if len(shuffled_features) > NUM_SEATS * chunk_size:
    seat_features[-1].extend(shuffled_features[NUM_SEATS * chunk_size:])

# Tracking dictionary for Hit Rate (Quarantine)
hit_rates_spy = {s: [] for s in range(NUM_SEATS)}
hit_rates_tlt = {s: [] for s in range(NUM_SEATS)}
hit_rates_gld = {s: [] for s in range(NUM_SEATS)}
pred_history = {} 

quarantine_counts = 0
total_seat_evals = 0

for i in rebalance_idx:
    current_date = data.index[i]
    X_current = data[feature_cols].iloc[i:i+1]
    
    # Resolve past predictions
    for s in range(NUM_SEATS):
        if s < 8: h = 1
        elif s < 16: h = 2
        elif s < 24: h = 3
        elif s < 32: h = 5
        elif s < 40: h = 10
        else: h = 21
        past_i = i - h
        if past_i in pred_history:
            past_preds = pred_history[past_i]
            t_str = f'{h}d'
            actual_spy = (data[f'SPY_target{t_str}'].iloc[past_i] > 0)
            actual_tlt = (data[f'TLT_target{t_str}'].iloc[past_i] > 0)
            actual_gld = (data[f'GLD_target{t_str}'].iloc[past_i] > 0)
            
            if s in past_preds['spy']:
                hit_rates_spy[s].append(int(past_preds['spy'][s] == actual_spy))
                hit_rates_tlt[s].append(int(past_preds['tlt'][s] == actual_tlt))
                hit_rates_gld[s].append(int(past_preds['gld'][s] == actual_gld))
                hit_rates_spy[s] = hit_rates_spy[s][-50:]
                hit_rates_tlt[s] = hit_rates_tlt[s][-50:]
                hit_rates_gld[s] = hit_rates_gld[s][-50:]

    spy_bull_votes = 0
    tlt_bull_votes = 0
    gld_bull_votes = 0
    
    spy_active_seats = 0
    tlt_active_seats = 0
    gld_active_seats = 0
    
    current_preds_spy = {}
    current_preds_tlt = {}
    current_preds_gld = {}
    
    for s in range(NUM_SEATS):
        feats = seat_features[s]
        if s < 8: h = 1
        elif s < 16: h = 2
        elif s < 24: h = 3
        elif s < 32: h = 5
        elif s < 40: h = 10
        else: h = 21
        train_end_idx_seat = i - h
        if train_end_idx_seat < TRAIN_WINDOW: continue
        
        train_slice_s = data.iloc[:train_end_idx_seat].dropna(subset=[f'SPY_target{h}d'] + feats)
        if len(train_slice_s) < 100: continue
        
        y_spy = (train_slice_s[f'SPY_target{h}d'] > 0).astype(int)
        y_tlt = (train_slice_s[f'TLT_target{h}d'] > 0).astype(int)
        y_gld = (train_slice_s[f'GLD_target{h}d'] > 0).astype(int)
        
        def fit_predict(X_tr, y_tr, X_curr):
            if y_tr.nunique() == 1:
                return y_tr.iloc[0]
            m = LGBMClassifier(n_estimators=10, max_depth=3, random_state=RANDOM_STATE+s, verbose=-1, n_jobs=1)
            m.fit(X_tr, y_tr)
            return m.predict(X_curr)[0]
            
        p_spy = fit_predict(train_slice_s[feats], y_spy, X_current[feats])
        p_tlt = fit_predict(train_slice_s[feats], y_tlt, X_current[feats])
        p_gld = fit_predict(train_slice_s[feats], y_gld, X_current[feats])
        
        current_preds_spy[s] = p_spy
        current_preds_tlt[s] = p_tlt
        current_preds_gld[s] = p_gld
        
        quar_spy = (np.mean(hit_rates_spy[s][-30:]) < 0.49) if len(hit_rates_spy[s]) >= 20 else False
        quar_tlt = (np.mean(hit_rates_tlt[s][-30:]) < 0.49) if len(hit_rates_tlt[s]) >= 20 else False
        quar_gld = (np.mean(hit_rates_gld[s][-30:]) < 0.49) if len(hit_rates_gld[s]) >= 20 else False
        
        total_seat_evals += 3
        if quar_spy: quarantine_counts += 1
        else:
            spy_active_seats += 1
            spy_bull_votes += p_spy
            
        if quar_tlt: quarantine_counts += 1
        else:
            tlt_active_seats += 1
            tlt_bull_votes += p_tlt
            
        if quar_gld: quarantine_counts += 1
        else:
            gld_active_seats += 1
            gld_bull_votes += p_gld

    pred_history[i] = {'spy': current_preds_spy, 'tlt': current_preds_tlt, 'gld': current_preds_gld}
    
    align_spy = spy_bull_votes / max(1, spy_active_seats)
    align_tlt = tlt_bull_votes / max(1, tlt_active_seats)
    align_gld = gld_bull_votes / max(1, gld_active_seats)
    
    alignments = {'SPY': align_spy, 'TLT': align_tlt, 'GLD': align_gld}
    best_asset = max(alignments, key=alignments.get)
    best_align = alignments[best_asset]
    
    # Fractional Confidence Logic
    if best_align < 0.30:
        pos = best_asset
        target_allocation = 0.0
    elif best_align < 0.50:
        pos = best_asset
        target_allocation = TARGET_VOL / 2.0
    else:
        pos = best_asset
        target_allocation = TARGET_VOL
        
    asset_vol = returns[pos].iloc[i - VOL_LOOKBACK + 1 : i + 1].std() * np.sqrt(252)
    asset_vol = max(asset_vol, 0.0001)
    
    if target_allocation == 0.0:
        weight = 0.0
    else:
        weight = min(target_allocation / asset_vol, MAX_EXPOSURE)
    
    # Portfolio period return
    asset_log = returns[pos].iloc[i+1 : i+1+REBALANCE_DAYS].sum()
    asset_simple = np.exp(asset_log) - 1
    cash_log = np.log(1 + rf_daily.iloc[i+1 : i+1+REBALANCE_DAYS]).sum()
    cash_simple = np.exp(cash_log) - 1
    port_simple = weight * asset_simple + (1 - weight) * cash_simple
    
    prev_pos = positions[-1] if positions else None
    prev_weight = weights[-1] if weights else 0.0
    if pos != prev_pos:
        turnover = prev_weight + weight
    else:
        turnover = abs(weight - prev_weight)
    port_simple -= turnover * TC
    
    port_log = np.log(max(1 + port_simple, 1e-10))
    
    positions.append(pos)
    weights.append(weight)
    strategy_period_rets.append(port_log)
    end_date = min(i + REBALANCE_DAYS, len(returns)-1)
    dates.append(returns.index[end_date])

# ====================== 4. PERFORMANCE ======================
strategy_rets = pd.Series(strategy_period_rets, index=dates)
strategy_cum = np.exp(strategy_rets.cumsum())
weights_series = pd.Series(weights, index=dates)

spy_cum = np.exp(returns['SPY'].loc[start_exec_date:].cumsum()).reindex(dates, method='ffill')
tlt_cum = np.exp(returns['TLT'].loc[start_exec_date:].cumsum()).reindex(dates, method='ffill')
gld_cum = np.exp(returns['GLD'].loc[start_exec_date:].cumsum()).reindex(dates, method='ffill')

def calc_metrics(cum):
    rets = cum.pct_change().dropna()
    total = cum.iloc[-1] / cum.iloc[0] - 1
    years = (cum.index[-1] - cum.index[0]).days / 365.25
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    sharpe = rets.mean() / rets.std() * np.sqrt(252 / REBALANCE_DAYS) if rets.std() != 0 else 0
    maxdd = ((cum / cum.cummax()) - 1).min()
    return {'CAGR (%)': round(cagr*100, 2), 'Sharpe': round(sharpe, 2), 'MaxDD (%)': round(maxdd*100, 2)}

print("\n=== THE SENATE RESULTS ===")
print("Strategy (Fractional Sizing):", calc_metrics(strategy_cum))
print("SPY Buy&Hold:", calc_metrics(spy_cum))
print("TLT Buy&Hold:", calc_metrics(tlt_cum))
print("GLD Buy&Hold:", calc_metrics(gld_cum))
print(f"Average deployed capital: {weights_series.mean()*100:.1f}% (rest in T-Bills)")
print(f"Quarantine Rate: {quarantine_counts / max(1, total_seat_evals)*100:.1f}% of seat-votes were vetoed due to poor moving accuracy.")

year_df = pd.DataFrame({'strategy': strategy_cum, 'spy': spy_cum})
yearly_end = year_df.groupby(year_df.index.year).last()
yearly_end_prev = yearly_end.shift(1).fillna(1.0)
yearly_returns = (yearly_end / yearly_end_prev) - 1

print("\n=== YEAR-BY-YEAR PERFORMANCE ===")
print(f"{'Year':<6} | {'Strategy%':>9} | {'SPY%':>9} | {'Strat Cum%':>10} | {'SPY Cum%':>10}")
print("-" * 55)
for y in yearly_end.index:
    strat_y = yearly_returns.loc[y, 'strategy'] * 100
    spy_y = yearly_returns.loc[y, 'spy'] * 100
    strat_c = (yearly_end.loc[y, 'strategy'] - 1) * 100
    spy_c = (yearly_end.loc[y, 'spy'] - 1) * 100
    print(f"{y:<6} | {strat_y:>8.2f}% | {spy_y:>8.2f}% | {strat_c:>9.2f}% | {spy_c:>9.2f}%")

results_df = pd.DataFrame({
    'date': dates,
    'position': positions,
    'weight': weights,
    'strategy_cum': strategy_cum.values,
    'spy_cum': spy_cum.values,
})
results_df.to_csv('senate_backtest_results.csv', index=False)
print("📊 Output logic saved.")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
ax1.plot(strategy_cum, label='Senate Strategy', linewidth=2.5, color='royalblue')
ax1.plot(spy_cum, label='SPY', alpha=0.5, color='gray')
ax1.set_title('Senate Architecture Fractional Rotation vs SPY', fontweight='bold')
ax1.set_ylabel('Growth of $1')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.fill_between(weights_series.index, 0, weights_series*100, color='green', alpha=0.2)
ax2.plot(weights_series.index, weights_series*100, color='green', linewidth=1.5)
ax2.set_ylabel('Exposure %')
ax2.set_ylim(0, 110)
ax2.legend(['Capital Deployed'])
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('backtest_plot_senate_performance.png')
plt.close()
