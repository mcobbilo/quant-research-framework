import pandas as pd
import numpy as np
import requests
import datetime
import pandas_datareader.data as web

def fetch_fiscal_data():
    print("Fetching FiscalData for Treasury Issuance Skew...")
    # Fetch up to 10,000 records from mspd_table_1
    url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_1?page[size]=10000"
    response = requests.get(url).json()
    
    records = []
    for row in response.get("data", []):
        if row.get("security_type_desc") == "Marketable":
            cls_desc = row.get("security_class_desc", "")
            if cls_desc in ["Bills", "Notes", "Bonds"]:
                date_str = row.get("record_date")
                amt = float(row.get("total_mil_amt", 0).replace(",", ""))
                records.append({"Date": date_str, "Type": cls_desc, "Amount": amt})
                
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
        
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pivot to get Bills, Notes, Bonds as columns
    pivot = df.pivot_table(index="Date", columns="Type", values="Amount", aggfunc='sum').fillna(0)
    
    # Calculate ratio: Bills / (Bills + Notes + Bonds)
    pivot['Total_Marketable_Core'] = pivot['Bills'] + pivot['Notes'] + pivot['Bonds']
    pivot['t_bill_drain_ratio'] = pivot['Bills'] / pivot['Total_Marketable_Core']
    
    return pivot[['t_bill_drain_ratio']].reset_index()

def main():
    start_date = "1999-01-01"
    end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    
    print("Fetching Daily/Weekly FRED Telemetry...")
    # Daily features
    fred_series = {
        'DFF': 'Fed_Funds',
        'DGS2': 'US2Y',
        'BAMLH0A0HYM2': 'HY_Spread',
        'WALCL': 'Total_Assets',
        'WTREGEN': 'TGA',
        'RRPONTSYD': 'RRP',
        'CPILFENS': 'Core_CPI_NSA',
        'UNRATENSA': 'Unemployment_NSA'
    }
    
    df_fred = web.DataReader(list(fred_series.keys()), 'fred', start_date, end_date)
    df_fred.rename(columns=fred_series, inplace=True)
    
    # RRP was functionally 0 before the facility was expanded in the 2010s
    df_fred['RRP'] = df_fred['RRP'].fillna(0)
    
    # Forward fill missing daily spots for weekly/monthly metrics
    df_fred = df_fred.ffill()
    
    # 1. Taylor Rule Deviance
    # NSA CPI YoY Inflation
    df_fred['Core_CPI_NSA_YoY'] = df_fred['Core_CPI_NSA'].pct_change(252) * 100 # approx 1 yr trading days since it's daily ffilled
    
    # Taylor Rule = CPI_NSA_YoY + 2.0 (neutral) + 0.5*(CPI_NSA_YoY - 2.0) - 0.5*(UNRATE_NSA - 4.5)
    df_fred['Taylor_Rule_Rate'] = df_fred['Core_CPI_NSA_YoY'] + 2.0 + 0.5 * (df_fred['Core_CPI_NSA_YoY'] - 2.0) - 0.5 * (df_fred['Unemployment_NSA'] - 4.5)
    df_fred['taylor_policy_spread'] = df_fred['Fed_Funds'] - df_fred['Taylor_Rule_Rate']
    
    # 2. Smart Money Shadow
    df_fred['us2y_ffr_divergence'] = df_fred['US2Y'] - df_fred['Fed_Funds']
    
    # 3. Global Net Liquidity Velocity
    df_fred['Net_Liquidity'] = df_fred['Total_Assets'] - df_fred['TGA'] - df_fred['RRP']
    df_fred['net_liquidity_momentum'] = df_fred['Net_Liquidity'].pct_change(65) # 13 weeks * 5 days = 65 trading days
    
    # 4. Credit Cycle Peak / Malinvestment
    df_fred['hy_spread_velocity'] = df_fred['HY_Spread'].pct_change(21) # 1 month velocity
    
    df_fred = df_fred.reset_index()
    if 'DATE' in df_fred.columns: df_fred.rename(columns={'DATE': 'Date'}, inplace=True)
    
    # 5. Treasury Issuance Skew
    df_fiscal = fetch_fiscal_data()
    
    print("Fusing Telemetry Structures...")
    # Merge Fiscal data
    df_final = pd.merge(df_fred, df_fiscal, on='Date', how='left')
    
    # Forward fill the monthly treasury skew to daily
    df_final['t_bill_drain_ratio'] = df_final['t_bill_drain_ratio'].ffill()
    
    # ---------------------------------------------------------
    # 💥 RED TEAM HARDENING: LATENCY SHIFT 💥
    # Macroeconomic reports (CPI, Unemployment, Treasury Issuance)
    # usually drop 3-4 weeks after the month closes.
    # We enforce a strict 21-trading-day shift to eliminate lookahead bias.
    # ---------------------------------------------------------
    shift_cols = ['taylor_policy_spread', 'us2y_ffr_divergence', 'net_liquidity_momentum', 'hy_spread_velocity', 't_bill_drain_ratio']
    df_final[shift_cols] = df_final[shift_cols].shift(21)

    # Drop rows that don't have enough history to calc YoY arrays (first year)
    df_final = df_final.dropna(subset=['taylor_policy_spread', 'net_liquidity_momentum'])
    
    print("Variables Encoded:")
    print(df_final[['Date', 'taylor_policy_spread', 'us2y_ffr_divergence', 'net_liquidity_momentum', 'hy_spread_velocity', 't_bill_drain_ratio']].tail())
    
    # Output
    out_cols = ['Date', 'taylor_policy_spread', 'us2y_ffr_divergence', 'net_liquidity_momentum', 'hy_spread_velocity', 't_bill_drain_ratio']
    df_final[out_cols].to_csv("druckenmiller_features.csv", index=False)
    print(f"\nPipeline Complete. Saved to druckenmiller_features.csv ({len(df_final)} rows)")

if __name__ == "__main__":
    main()
