# universal_fyers_scraper.py
#
# Description:
# This is a consolidated, universal data scraper for the Nifty 200 project.
# It has been upgraded to include the intelligent features from the original
# scrapers, such as incremental updates and robust error handling.
#
# Functionality:
# - Correctly authenticates using the fyers_apiv3 library and a long-lived access token.
# - Intelligently updates data by fetching only new records since the last download.
# - Provides a --force flag to re-download all data from the beginning.
# - Scrapes Daily and 15-Minute historical data for both Equities and Indices.
# - Saves all data to a single, unified directory: 'data/historical_data/'.
#
# Usage:
#   - To intelligently update all data (default):
#     python universal_fyers_scraper.py
#
#   - To force a full re-download of all data:
#     python universal_fyers_scraper.py --force
#
#   - To update only a specific interval (e.g., daily):
#     python universal_fyers_scraper.py --interval daily

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import time
import json

# --- Import Configuration ---
# Imports credentials (CLIENT_ID, SECRET_KEY, etc.) from a local config.py file.
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
    "output_dir": os.path.join("data", "historical_data"),
    "nifty_list_csv": "nifty200.csv",
    "index_list": ["NIFTY200_INDEX", "INDIAVIX"],
    "default_start_date": "2020-01-01",
    "token_file": "fyers_access_token.txt",
    "log_path": os.getcwd(),
    "api_cooldown_seconds": 1.1,
    "api_retries": 3,
}

def get_access_token():
    """
    Handles the Fyers authentication flow to get the access token.
    Reads from a file if available, otherwise generates a new one via user input.
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
    Fetches historical OHLC data using the fyers_apiv3 client with a retry mechanism.
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
            if response.get("s") == 'ok':
                candles = response.get('candles', [])
                if not candles:
                    return pd.DataFrame()
                
                df = pd.DataFrame(candles, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                # Fyers API provides data in UTC, convert to IST for consistency
                df['datetime'] = df['datetime'] + pd.Timedelta(hours=5, minutes=30)
                df.set_index('datetime', inplace=True)
                return df
            else:
                print(f"    - API Error (Attempt {i+1}): {response.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"    - An exception occurred (Attempt {i+1}): {e}")
        
        if i < SCRIPT_CONFIG["api_retries"] - 1:
            time.sleep(SCRIPT_CONFIG["api_cooldown_seconds"] * 2) # Longer delay on retry

    print(f"    - Failed to fetch data for {symbol} after {SCRIPT_CONFIG['api_retries']} attempts.")
    return pd.DataFrame()

def scrape_data_for_symbols(fyers_client, symbol_list, interval, file_suffix, force_download):
    """
    Iterates through symbols, intelligently fetching and updating data.
    """
    print(f"\n--- Starting Scrape for {len(symbol_list)} symbols | Interval: {interval} ---")
    resolution = "D" if interval == "daily" else "15"

    for i, symbol_name in enumerate(symbol_list):
        print(f"\nProcessing {symbol_name} ({i+1}/{len(symbol_list)})...")
        
        output_path = os.path.join(SCRIPT_CONFIG["output_dir"], f"{symbol_name}_{file_suffix}.csv")
        to_date = datetime.now()
        
        from_date = None
        if force_download:
            from_date = pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date()
            print(f"  > FORCE DOWNLOAD enabled. Fetching all data from {from_date.strftime('%Y-%m-%d')}")
        elif os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                if not existing_df.empty:
                    last_date_str = existing_df['datetime'].iloc[-1]
                    last_date = pd.to_datetime(last_date_str).date()
                    from_date = last_date + timedelta(days=1)
                    print(f"  > Existing data found. Fetching new data from {from_date.strftime('%Y-%m-%d')}")
                else:
                    from_date = pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date()
                    print(f"  > Existing file is empty. Performing full download.")
            except Exception as e:
                from_date = pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date()
                print(f"  > Could not read existing file. Performing full download. Error: {e}")
        else:
            from_date = pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date()
            print(f"  > No existing data. Performing full download.")

        if from_date > to_date.date():
             print(f"  > Data is already up to date. Skipping.")
             continue
        
        all_data_batches = []
        batch_start_date = from_date

        while batch_start_date <= to_date.date():
            batch_end_dt = pd.to_datetime(batch_start_date) + pd.DateOffset(months=6) - timedelta(days=1)
            if batch_end_dt.date() > to_date.date():
                batch_end_dt = to_date

            from_date_str = batch_start_date.strftime('%Y-%m-%d')
            to_date_str = batch_end_dt.strftime('%Y-%m-%d')
            
            print(f"    > Fetching batch: {from_date_str} to {to_date_str}")
            fyers_symbol = f"NSE:{symbol_name}-EQ" if symbol_name not in SCRIPT_CONFIG["index_list"] else f"NSE:{symbol_name}"
            batch_df = get_historical_data(fyers_client, fyers_symbol, resolution, from_date_str, to_date_str)
            
            if not batch_df.empty:
                all_data_batches.append(batch_df)

            time.sleep(SCRIPT_CONFIG["api_cooldown_seconds"])
            batch_start_date = batch_end_dt.date() + timedelta(days=1)

        if not all_data_batches:
            print(f"  > Info: No new data returned for {symbol_name}.")
            continue
            
        new_data_df = pd.concat(all_data_batches)
        if force_download or not os.path.exists(output_path):
            new_data_df.to_csv(output_path, mode='w', header=True, index_label='datetime')
            print(f"  > Success: Saved {len(new_data_df)} records to {output_path}")
        else:
            # Append new data without writing the header
            new_data_df.to_csv(output_path, mode='a', header=False)
            print(f"  > Success: Appended {len(new_data_df)} new records to {output_path}")

def main():
    """
    Main function to orchestrate the scraping process.
    """
    parser = argparse.ArgumentParser(description="Universal Fyers Scraper for Nifty 200 Project.")
    parser.add_argument('--interval', type=str, required=False, choices=['daily', '15min'], help='Specify an interval to scrape. If not provided, all intervals will be scraped.')
    parser.add_argument('--force', action='store_true', help='Force a full re-download of data, ignoring existing files.')
    args = parser.parse_args()

    # --- Fyers Client Initialization ---
    access_token = get_access_token()
    if not access_token:
        print("Could not obtain access token. Exiting.")
        sys.exit(1)

    try:
        fyers = fyersModel.FyersModel(
            client_id=config.CLIENT_ID,
            is_async=False,
            token=access_token,
            log_path=SCRIPT_CONFIG["log_path"]
        )
        print("Fyers API client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Fyers API client: {e}")
        sys.exit(1)
    
    # --- Setup ---
    os.makedirs(SCRIPT_CONFIG["output_dir"], exist_ok=True)
    print(f"Output directory set to: '{SCRIPT_CONFIG['output_dir']}'")
    
    try:
        equity_symbols = pd.read_csv(SCRIPT_CONFIG["nifty_list_csv"])["Symbol"].tolist()
    except FileNotFoundError:
        print(f"FATAL ERROR: Symbol file '{SCRIPT_CONFIG['nifty_list_csv']}' not found. Exiting.")
        sys.exit(1)
        
    index_symbols = SCRIPT_CONFIG["index_list"]

    # --- Execution Logic ---
    run_daily = not args.interval or args.interval == 'daily'
    run_15min = not args.interval or args.interval == '15min'

    if run_daily:
        scrape_data_for_symbols(fyers, equity_symbols, "daily", "daily", args.force)
        scrape_data_for_symbols(fyers, index_symbols, "daily", "daily", args.force)
    
    if run_15min:
        scrape_data_for_symbols(fyers, equity_symbols, "15min", "15min", args.force)
        scrape_data_for_symbols(fyers, index_symbols, "15min", "15min", args.force)

    print("\n--- Universal scraping process complete! ---")

if __name__ == "__main__":
    main()
