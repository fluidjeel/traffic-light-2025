import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import time

# --- Import Configuration ---
try:
    from config import CLIENT_ID, SECRET_KEY, REDIRECT_URI
except ImportError:
    print("Error: Could not import from config.py.")
    sys.exit()

from fyers_apiv3 import fyersModel

# --- Script Configuration ---
LOG_PATH = os.getcwd()
TOKEN_FILE = "fyers_access_token.txt"
START_DATE = "2020-01-01"
MAPPING_FILE = 'stock_to_sector.csv'
OUTPUT_DIR = "historical_data"

def get_access_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f: return f.read().strip()
    else:
        print(f"Error: {TOKEN_FILE} not found. Please run your main scraper first.")
        return None

access_token = get_access_token()
if not access_token: sys.exit()

try:
    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=access_token, log_path=LOG_PATH)
    print("Fyers API client initialized successfully.")
except Exception as e:
    print(f"Error initializing Fyers API client: {e}")
    sys.exit()

def get_historical_data(symbol, resolution, from_date, to_date, retries=3, delay=2):
    data = {"symbol": symbol, "resolution": resolution, "date_format": "1", "range_from": from_date, "range_to": to_date, "cont_flag": "1"}
    for i in range(retries):
        try:
            response = fyers.history(data=data)
            if response.get("s") == 'ok':
                candles = response.get('candles', [])
                if not candles: return pd.DataFrame()
                df = pd.DataFrame(candles, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s') + pd.Timedelta(hours=5, minutes=30)
                df.set_index('datetime', inplace=True)
                return df
            else:
                print(f"    - API Error for {symbol} (Attempt {i+1}/{retries}): {response.get('message', 'Unknown error')}")
                if i < retries - 1: time.sleep(delay)
        except Exception as e:
            print(f"    - Exception for {symbol} (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1: time.sleep(delay)
    return pd.DataFrame()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        mapping_df = pd.read_csv(MAPPING_FILE)
        unique_sectors = mapping_df['sector_symbol'].unique()
        print(f"Found {len(unique_sectors)} unique sector indices to download.")
    except FileNotFoundError:
        print(f"Error: Mapping file not found at '{MAPPING_FILE}'. Please run 'create_sector_mapping.py' first.")
        sys.exit()

    for sector_symbol_name in unique_sectors:
        fyers_symbol = f"NSE:{sector_symbol_name}-INDEX"
        output_filename = f"{sector_symbol_name}_INDEX_daily.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"\nFetching data for {sector_symbol_name}...")
        
        all_data_batches = []
        from_date = pd.to_datetime(START_DATE).date()
        to_date = datetime.now()

        batch_start_date = from_date
        while batch_start_date <= to_date.date():
            batch_end_dt = pd.to_datetime(batch_start_date) + pd.DateOffset(months=6) - timedelta(days=1)
            if batch_end_dt > to_date: batch_end_dt = to_date
            from_date_str = batch_start_date.strftime('%Y-%m-%d')
            to_date_str = batch_end_dt.strftime('%Y-%m-%d')
            
            print(f"    > Fetching batch: {from_date_str} to {to_date_str}")
            batch_df = get_historical_data(fyers_symbol, "D", from_date_str, to_date_str)
            if not batch_df.empty: all_data_batches.append(batch_df)
            time.sleep(1.2) # Increase delay slightly to be safe with API limits
            batch_start_date = batch_end_dt.date() + timedelta(days=1)

        if all_data_batches:
            final_df = pd.concat(all_data_batches)
            final_df.to_csv(output_path, mode='w', header=True)
            print(f"  > Success: Saved {len(final_df)} records to {output_path}")
        else:
            print(f"  > Info: No data returned for {sector_symbol_name}. It may be an invalid or delisted index.")
            
    print("\n--- Sector index data download complete! ---")
