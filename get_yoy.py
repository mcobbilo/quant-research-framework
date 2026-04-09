import pandas as pd
from red_team_xgboost_126_audit import run_xgboost_audit

# We will patch run_xgboost_audit to return df_cont as well
with open('red_team_xgboost_126_audit.py', 'r') as f:
    content = f.read()

content = content.replace("return final_ret, mdd", "return final_ret, mdd, df_cont")

with open('red_team_xgboost_126_audit_patched.py', 'w') as f:
    f.write(content)

from red_team_xgboost_126_audit_patched import run_xgboost_audit

final_ret, mdd, df_cont = run_xgboost_audit(seed=0, noise_std=0.0, guillotine=False, test_name="YEARLY_CALC")

df_cont['year'] = df_cont['ds'].dt.year
print("\n=== Year-by-Year Performance (Baseline) ===")
print(f"{'Year':<6} | {'Model Return':<14} | {'S&P 500 Return':<14} | {'Model Max DD':<14}")
print("-" * 55)

for year, group in df_cont.groupby('year'):
    comp_strat = (1 + group['strategy_return']).prod() - 1
    comp_spy = (1 + group['asset_return']).prod() - 1
    
    cum = (1 + group['strategy_return']).cumprod()
    peak = cum.expanding(min_periods=1).max()
    dd = ((cum / peak) - 1).min()
    
    print(f"{year:<6} | {comp_strat*100:>13.2f}% | {comp_spy*100:>13.2f}% | {dd*100:>13.2f}%")

print("===========================================")
print(f"Worst Drawdown over entire 10-Year Period: {mdd*100:.2f}%")

