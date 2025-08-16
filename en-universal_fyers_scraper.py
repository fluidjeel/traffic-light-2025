# universal_fyers_scraper.py (Enhanced for specific error handling and throttling)
#
# Description:
# This version of the scraper is enhanced to specifically handle the
# "Invalid symbol provided" and "request limit reached" errors. It will
# not retry a stock if an invalid symbol error occurs and will apply a
# longer cooldown for throttling errors. It also allows for explicit
# control over the number of parallel processes.
#
# MODIFICATION (v2.2 - Historical Backfill Logic):
# - Default start date changed to 2010-01-01 to fetch a longer history.
# - Refined the retry logic to explicitly use exponential backoff ONLY for
#   rate-limit errors, while using a fixed cooldown for other general API errors.
# - Added logic to intelligently backfill missing historical data for existing files.

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from multiprocessing import Pool, Manager, current_process
import numpy as np

# --- Import Configuration ---
try:
    import config
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! FATAL ERROR: config.py NOT FOUND                        !!!")
    print("!!! Please create a 'config.py' file with your API keys.    !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sys.exit(1)

# --- Import Fyers API Module ---
try:
    from fyers_apiv3 import fyersModel
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! FATAL ERROR: fyers_apiv3 library NOT FOUND              !!!")
    print("!!! Please install it using: pip install fyers-apiv3        !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sys.exit(1)

# --- SCRIPT-SPECIFIC SETTINGS ---
SCRIPT_CONFIG = {
    "output_dir": os.path.join("data", "universal_historical_data"),
    "nifty_list_csv": "nifty500.csv",
    "index_list": ["NIFTY200_INDEX", "INDIAVIX","NIFTY500-INDEX"],
    "default_start_date": "2010-01-01", # MODIFIED: Start date extended to 2010
    "token_file": "fyers_access_token.txt",
    "log_path": os.getcwd(),
    "api_cooldown_seconds": 1.1,
    "api_retries": 5, # Increased retries for better resilience
    "parallel_processes": 4 
}

# --- SYMBOL MAPPING FOR INDICES ---
FYERS_INDEX_SYMBOLS = {
    "NIFTY200_INDEX": "NSE:NIFTY200-INDEX",
    "INDIAVIX": "NSE:INDIAVIX-INDEX",
    "NIFTY500-INDEX": "NSE:NIFTY500-INDEX"
}

# --- GLOBAL VARIABLES FOR MULTIPROCESSING ---
global_fyers_client = None
global_interval = None
global_file_suffix = None
global_force_download = None
global_failed_symbols_list = None
global_throttled_symbols_list = None

# Unique sentinel values for specific errors
INVALID_SYMBOL_ERROR = "Invalid symbol provided"
THROTTLED_ERROR = "Request limit reached"

def worker_init(access_token, interval, file_suffix, force_download, failed_symbols_list, throttled_symbols_list):
    """
    Initializes the Fyers client and global settings for each worker process.
    """
    global global_fyers_client, global_interval, global_file_suffix, global_force_download, global_failed_symbols_list, global_throttled_symbols_list
    try:
        global_fyers_client = fyersModel.FyersModel(
            client_id=config.CLIENT_ID,
            is_async=False,
            token=access_token,
            log_path=SCRIPT_CONFIG["log_path"]
        )
        global_interval = interval
        global_file_suffix = file_suffix
        global_force_download = force_download
        global_failed_symbols_list = failed_symbols_list
        global_throttled_symbols_list = throttled_symbols_list
    except Exception as e:
        print(f"Worker process {current_process().pid} failed to initialize Fyers client: {e}")
        global_fyers_client = None

def get_access_token():
    """
    Handles the Fyers authentication flow to get the access token.
    """
    if os.path.exists(SCRIPT_CONFIG["token_file"]):
        with open(SCRIPT_CONFIG["token_file"], 'r') as f:
            return f.read().strip()

    session = fyersModel.SessionModel(
        client_id=config.CLIENT_ID,
        secret_key=config.SECRET_KEY,
        redirect_uri=config.REDIRECT_URI,
        response_type='code',
        grant_type='authorization_code'
    )
    
    auth_url = session.generate_authcode()
    print("--- Fyers Login Required ---")
    print(f"1. Go to this URL and log in: {auth_url}")
    print("2. After logging in, you will be redirected to a blank page.")
    print("3. Copy the 'auth_code' from the redirected URL's address bar.")
    
    auth_code = input("4. Enter the auth_code here: ")

    session.set_token(auth_code)
    response = session.generate_token()

    if response.get("access_token"):
        access_token = response["access_token"]
        with open(SCRIPT_CONFIG["token_file"], 'w') as f:
            f.write(access_token)
        print("Access token generated and saved successfully.")
        return access_token
    else:
        print(f"Failed to generate access token: {response}")
        return None

def get_historical_data(fyers_client, symbol, resolution, from_date, to_date):
    """
    Fetches historical OHLC data with a retry mechanism and exponential backoff.
    """
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": from_date,
        "range_to": to_date,
        "cont_flag": "1"
    }
    
    for i in range(SCRIPT_CONFIG["api_retries"]):
        try:
            response = fyers_client.history(data=data)
            
            if response.get("message") == "Invalid symbol provided":
                print(f"    - Worker {current_process().pid} ERROR: Invalid symbol {symbol}. Skipping retries.")
                return INVALID_SYMBOL_ERROR
            
            # Exponential backoff specifically for rate limiting
            if "request limit reached" in response.get("message", "").lower():
                wait_time = (2 ** i) + np.random.uniform(0, 1) # Add jitter
                print(f"    - Worker {current_process().pid} INFO: API rate limit reached for {symbol}. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time) 
                continue 
            
            if response.get("s") == 'ok':
                candles = response.get('candles', [])
                if not candles:
                    return pd.DataFrame()
                
                df = pd.DataFrame(candles, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                df['datetime'] = df['datetime'] + pd.Timedelta(hours=5, minutes=30)
                df.set_index('datetime', inplace=True)
                return df
            else:
                # Use a fixed delay for other general API errors
                print(f"    - Worker {current_process().pid} API Error for {symbol} (Attempt {i+1}): {response.get('message', 'Unknown error')}")
                time.sleep(SCRIPT_CONFIG["api_cooldown_seconds"])

        except Exception as e:
            # Use a fixed delay for exceptions
            print(f"    - Worker {current_process().pid} An exception occurred for {symbol} (Attempt {i+1}): {e}")
            time.sleep(SCRIPT_CONFIG["api_cooldown_seconds"])

    print(f"    - Worker {current_process().pid} Failed to fetch data for {symbol} after {SCRIPT_CONFIG['api_retries']} attempts.")
    return THROTTLED_ERROR # Return error if all retries fail

def process_single_symbol(symbol_name):
    """
    Worker function to scrape data for a single symbol, with backfill logic.
    """
    global global_fyers_client, global_interval, global_file_suffix, global_force_download, global_failed_symbols_list, global_throttled_symbols_list
    
    if not global_fyers_client:
        print(f"Worker process {current_process().pid} has no Fyers client. Skipping {symbol_name}.")
        return

    print(f"Worker {current_process().pid} processing {symbol_name}...")
    
    resolution = "D" if global_interval == "daily" else "15"
    batch_months = 6 if global_interval == "daily" else 2
    output_path = os.path.join(SCRIPT_CONFIG["output_dir"], f"{symbol_name}_{global_file_suffix}.csv")
    
    existing_df = None
    if os.path.exists(output_path) and not global_force_download:
        try:
            existing_df = pd.read_csv(output_path, index_col='datetime', parse_dates=True)
        except Exception as e:
            print(f"    - Worker {current_process().pid} Could not read existing file. Performing full download. Error: {e}")
            existing_df = None

    # --- Backfill Logic ---
    all_backfill_batches = []
    if existing_df is not None and not existing_df.empty:
        first_date_in_file = existing_df.index.min().date()
        backfill_start_date = pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date()
        backfill_end_date = first_date_in_file - timedelta(days=1)

        if backfill_start_date < backfill_end_date:
            print(f"    - Worker {current_process().pid} Backfilling data from {backfill_start_date} to {backfill_end_date}")
            batch_start_date = backfill_start_date
            while batch_start_date <= backfill_end_date:
                batch_end_dt = pd.to_datetime(batch_start_date) + pd.DateOffset(months=batch_months) - timedelta(days=1)
                if batch_end_dt.date() > backfill_end_date:
                    batch_end_dt = pd.to_datetime(backfill_end_date)

                from_date_str = batch_start_date.strftime('%Y-%m-%d')
                to_date_str = batch_end_dt.strftime('%Y-%m-%d')
                
                fyers_symbol = FYERS_INDEX_SYMBOLS.get(symbol_name, f"NSE:{symbol_name}-EQ")
                
                print(f"    - Worker {current_process().pid} Fetching backfill batch for {symbol_name}: {from_date_str} to {to_date_str}")
                batch_df = get_historical_data(global_fyers_client, fyers_symbol, resolution, from_date_str, to_date_str)
                
                if isinstance(batch_df, str):
                    if batch_df == INVALID_SYMBOL_ERROR: global_failed_symbols_list.append(symbol_name); return
                    if batch_df == THROTTLED_ERROR: global_throttled_symbols_list.append(symbol_name); return
                if not batch_df.empty: all_backfill_batches.append(batch_df)
                batch_start_date = batch_end_dt.date() + timedelta(days=1)

    # --- Forward Fill (Append) Logic ---
    all_forward_batches = []
    to_date = datetime.now()
    from_date = pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date()
    
    if existing_df is not None and not existing_df.empty:
        from_date = existing_df.index.max().date() + timedelta(days=1)

    if from_date <= to_date.date():
        print(f"    - Worker {current_process().pid} Fetching new data from {from_date}")
        batch_start_date = from_date
        while batch_start_date <= to_date.date():
            batch_end_dt = pd.to_datetime(batch_start_date) + pd.DateOffset(months=batch_months) - timedelta(days=1)
            if batch_end_dt.date() > to_date.date():
                batch_end_dt = to_date

            from_date_str = batch_start_date.strftime('%Y-%m-%d')
            to_date_str = batch_end_dt.strftime('%Y-%m-%d')
            
            fyers_symbol = FYERS_INDEX_SYMBOLS.get(symbol_name, f"NSE:{symbol_name}-EQ")
            
            print(f"    - Worker {current_process().pid} Fetching forward batch for {symbol_name}: {from_date_str} to {to_date_str}")
            batch_df = get_historical_data(global_fyers_client, fyers_symbol, resolution, from_date_str, to_date_str)
            
            if isinstance(batch_df, str):
                if batch_df == INVALID_SYMBOL_ERROR: global_failed_symbols_list.append(symbol_name); return
                if batch_df == THROTTLED_ERROR: global_throttled_symbols_list.append(symbol_name); return
            if not batch_df.empty: all_forward_batches.append(batch_df)
            batch_start_date = batch_end_dt.date() + timedelta(days=1)

    # --- Combine and Save ---
    if not all_backfill_batches and not all_forward_batches:
        print(f"    - Worker {current_process().pid} No new or historical data to save for {symbol_name}.")
        return

    backfill_df = pd.concat(all_backfill_batches) if all_backfill_batches else pd.DataFrame()
    forward_df = pd.concat(all_forward_batches) if all_forward_batches else pd.DataFrame()
    
    # Combine all dataframes: backfilled, existing, and new forward data
    combined_df = pd.concat([backfill_df, existing_df, forward_df])
    
    # Remove any duplicates and sort chronologically
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    combined_df.sort_index(inplace=True)
    
    # Overwrite the file with the complete dataset
    combined_df.to_csv(output_path, mode='w', header=True, index_label='datetime')
    print(f"    - Worker {current_process().pid} Success: Saved/updated {len(combined_df)} total records for {symbol_name} to {output_path}")

def main():
    """
    Main function to orchestrate the scraping process.
    """
    parser = argparse.ArgumentParser(description="Universal Fyers Scraper.")
    parser.add_argument('--interval', type=str, required=False, choices=['daily', '15min'], help='Specify an interval to scrape. If not provided, all intervals will be scraped.')
    parser.add_argument('--force', action='store_true', help='Force a full re-download of data, ignoring existing files.')
    parser.add_argument('--only-index', action='store_true', help='Scrape data only for the indices defined in the config.')
    args = parser.parse_args()

    access_token = get_access_token()
    if not access_token:
        print("Could not obtain access token. Exiting.")
        sys.exit(1)
    
    os.makedirs(SCRIPT_CONFIG["output_dir"], exist_ok=True)
    print(f"Output directory set to: '{SCRIPT_CONFIG['output_dir']}'")
    
    equity_symbols = []
    if args.only_index:
        print("\n--only-index flag detected. Scraping only index symbols.--")
    elif SCRIPT_CONFIG["nifty_list_csv"]:
        try:
            equity_symbols = pd.read_csv(SCRIPT_CONFIG["nifty_list_csv"])["Symbol"].tolist()
        except FileNotFoundError:
            print(f"WARNING: Equity file '{SCRIPT_CONFIG['nifty_list_csv']}' not found. Skipping equities.")
        except pd.errors.EmptyDataError:
             print(f"WARNING: Equity file '{SCRIPT_CONFIG['nifty_list_csv']}' is empty. Skipping equities.")
        
    index_symbols = SCRIPT_CONFIG["index_list"]

    run_daily = not args.interval or args.interval == 'daily'
    run_15min = not args.interval or args.interval == '15min'

    with Manager() as manager:
        failed_symbols = manager.list()
        throttled_symbols = manager.list()
        
        num_processes = SCRIPT_CONFIG["parallel_processes"]
        print(f"\nStarting parallel downloads using {num_processes} processes.")

        if run_15min:
            print(f"\n--- Starting 15-minute scrape for {len(equity_symbols)} equities... ---")
            with Pool(initializer=worker_init, initargs=(access_token, "15min", "15min", args.force, failed_symbols, throttled_symbols)) as pool:
                pool.map(process_single_symbol, equity_symbols)
            
            print(f"\n--- Starting 15-minute scrape for {len(index_symbols)} indices... ---")
            with Pool(initializer=worker_init, initargs=(access_token, "15min", "15min", args.force, failed_symbols, throttled_symbols)) as pool:
                pool.map(process_single_symbol, index_symbols)
        
        if run_daily:
            print(f"\n--- Starting daily scrape for {len(equity_symbols)} equities... ---")
            with Pool(initializer=worker_init, initargs=(access_token, "daily", "daily", args.force, failed_symbols, throttled_symbols)) as pool:
                pool.map(process_single_symbol, equity_symbols)
                
            print(f"\n--- Starting daily scrape for {len(index_symbols)} indices... ---")
            with Pool(initializer=worker_init, initargs=(access_token, "daily", "daily", args.force, failed_symbols, throttled_symbols)) as pool:
                pool.map(process_single_symbol, index_symbols)
    
        print("\n--- Universal scraping process complete! ---")
        
        if failed_symbols:
            print("\n!!! The following symbols failed due to an 'Invalid symbol' error: !!!")
            for symbol in sorted(list(set(failed_symbols))):
                print(f"- {symbol}")
            print("\nPlease correct these symbols and try again.")
        
        if throttled_symbols:
            print("\n!!! The following symbols were skipped due to API throttling: !!!")
            for symbol in sorted(list(set(throttled_symbols))):
                print(f"- {symbol}")
            print("\nThese symbols can be retried later.")

if __name__ == "__main__":
    main()
