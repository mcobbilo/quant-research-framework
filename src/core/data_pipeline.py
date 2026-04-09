import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import logging

# Ensure correct pathing when running from root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.xlstm_wrapper import xLSTMForecast

logging.basicConfig(level=logging.INFO)

def spin_up_pipeline():
    logging.info("[Pipeline] Spinning up Core Data Pipeline for xLSTM...")
    
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(env_path)
    
    # Fetch Market Data
    logging.info("[Pipeline] Fetching SPY equity data...")
    ticker = yf.Ticker("SPY")
    df_spy = ticker.history(period="5y") # 5 years of daily data for performance
    df_spy = df_spy.reset_index()
    
    # Nixtla expects strictly: unique_id, ds, y
    df_spy.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    df_spy['ds'] = pd.to_datetime(df_spy['ds']).dt.tz_localize(None)
    df_spy['unique_id'] = 'SPY'
    
    # Engineer historical exogenous features
    df_spy['volume'] = df_spy['Volume']
    df_spy['high'] = df_spy['High']
    df_spy['low'] = df_spy['Low']
    
    # If FRED is available, we add macroeconomic data (T10Y2Y)
    fred_key = os.getenv("FRED_API_KEY")
    if fred_key:
        try:
            logging.info("[Pipeline] Fetching Macroeconomic Spreads via FRED API...")
            fred = Fred(api_key=fred_key)
            t10y2y = fred.get_series('T10Y2Y')
            t10y2y_df = pd.DataFrame({'ds': t10y2y.index, 't10y2y': t10y2y.values})
            t10y2y_df['ds'] = pd.to_datetime(t10y2y_df['ds']).dt.tz_localize(None)
            
            # Merge on closing date
            df_spy = pd.merge(df_spy, t10y2y_df, on='ds', how='left')
            df_spy['t10y2y'] = df_spy['t10y2y'].ffill().bfill() # Forward fill missing macro dates
            hist_exog = ['volume', 'high', 'low', 't10y2y']
            logging.info("[Pipeline] Macro data merged seamlessly.")
        except Exception as e:
            logging.error(f"[Pipeline] FRED fetch failed: {e}")
            hist_exog = ['volume', 'high', 'low']
    else:
        logging.warning("[Pipeline] FRED_API_KEY not found. Proceeding with price-action only.")
        hist_exog = ['volume', 'high', 'low']
        
    df_core = df_spy[['unique_id', 'ds', 'y'] + hist_exog].dropna().tail(1000)
    logging.info(f"[Pipeline] Cleaned Data aggregated: {len(df_core)} rows.")
    
    logging.info("[Pipeline] Initializing xLSTMForecast engine...")
    model = xLSTMForecast(
        h=5, # 5-day horizon
        input_size=15, # 15-day lookback window
        max_steps=10, # Strict steps for pipeline spin-up speed
        hist_exog_list=hist_exog,
        freq='B'
    )
    
    logging.info("[Pipeline] Commencing exponential gating fit...")
    model.fit(df_core)
    
    logging.info("[Pipeline] Fit complete. Running Point-Quantile Predictions...")
    predictions = model.predict(df_core)
    
    logging.info("[Pipeline] Pipeline execution finished successfully.")
    print("="*60)
    print(" xLSTM 5-DAY HORIZON PREDICTIONS (Quantiles: 0.1, 0.5, 0.9)")
    print("="*60)
    print(predictions.tail(5))
    print("="*60)

if __name__ == "__main__":
    spin_up_pipeline()
