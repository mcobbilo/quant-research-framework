import pandas as pd
import datetime

def fetch_net_dollar_liquidity(start_date="2020-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # FRED Series IDs:
    # WALCL: Total Assets
    # WTREGEN: Treasury General Account
    # RRPONTSYD: Reverse Repo Facility
    series = ['WALCL', 'WTREGEN', 'RRPONTSYD']
    
    # Fetch directly from FRED CSV endpoints with robust fallback
    try:
        import urllib.request
        import io
        dfs = []
        for s in series:
            req = urllib.request.Request(
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                csv_data = response.read().decode('utf-8')
            s_df = pd.read_csv(io.StringIO(csv_data), na_values='.')
            
            # Find the date column which might be DATE or observation_date
            date_col = 'DATE' if 'DATE' in s_df.columns else s_df.columns[0]
            s_df[date_col] = pd.to_datetime(s_df[date_col])
            s_df.set_index(date_col, inplace=True)
            
            s_df.rename(columns={s: s}, inplace=True)
            dfs.append(s_df)
        
        df = pd.concat(dfs, axis=1)
        df.index.name = 'DATE'
        
        # Filter by date exactly like the API
        df = df.loc[start_date:end_date]
        
        # Clean data 
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        # True North Float formulation
        df['Net_Dollar_Liquidity'] = df['WALCL'] - df['WTREGEN'] - df['RRPONTSYD']
        
        # Liquidity Derivative: 20-day ROC
        df['NDL_ROC_20D'] = df['Net_Dollar_Liquidity'].pct_change(20)
        
        # Binarize derivative for Guillotine
        df['Liquidity_Guillotine'] = df['NDL_ROC_20D'] > 0.0
        
        df.reset_index(inplace=True)
        df.rename(columns={'DATE': 'Date', 'index': 'Date'}, inplace=True)
        return df
    except Exception as e:
        print(f"Warning: Failed to fetch FRED liquidity telemetry -> {e}")
        return pd.DataFrame(columns=['Date', 'Liquidity_Guillotine', 'NDL_ROC_20D', 'Net_Dollar_Liquidity'])

if __name__ == "__main__":
    liquidity_df = fetch_net_dollar_liquidity()
    print("Net Dollar Liquidity Telemetry:")
    print(liquidity_df.tail())
