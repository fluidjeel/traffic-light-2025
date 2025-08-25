# universal_fyers_scraper.py (Enhanced for specific error handling, throttling, 1min timeframe support with fractional months fix, and optimized incremental download)
#
# Description:
# This scraper downloads historical OHLC data from Fyers API.
# It handles "Invalid symbol provided" and "request limit reached" errors specifically.
# Supports multiprocessing with control over parallel processes.
# Before making API calls, it checks the existing CSV files for each symbol and interval,
# determines the missing date ranges, and only fetches those missing intervals,
# minimizing redundant data downloads and API calls.
#
# MODIFICATION (Data Corruption Fix):
# - Removed 'set_index()' to keep 'datetime' as a regular column.
# - Added 'index=False' to all .to_csv() calls to prevent pandas from
#   writing the DataFrame index as extra columns.

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import time
from multiprocessing import Pool, Manager, current_process

# --- Import Configuration ---
try:
    import config
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! FATAL ERROR: config.py NOT FOUND !!!")
    print("!!! Please create a 'config.py' file with your API keys. !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sys.exit(1)

# --- Import Fyers API Module ---
try:
    from fyers_apiv3 import fyersModel
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! FATAL ERROR: fyers_apiv3 library NOT FOUND !!!")
    print("!!! Please install it using: pip install fyers-apiv3 !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sys.exit(1)

# --- SCRIPT-SPECIFIC SETTINGS ---
SCRIPT_CONFIG = {
    "output_dir": os.path.join("data", "universal_historical_data"),
    "nifty_list_csv": "nifty500.csv",
    "index_list": [
        "NIFTY200_INDEX",
        "INDIAVIX",
        "NIFTY500-INDEX",
        "NIFTY50-INDEX",
        "NIFTYBANK-INDEX",
        "GOLD25OCTFUT"
    ],
    "default_start_date": "2018-01-01",
    "token_file": "fyers_access_token.txt",
    "log_path": os.getcwd(),
    "api_cooldown_seconds": 1.1,
    "api_retries": 3,
    "parallel_processes": 4
}

# --- SYMBOL MAPPING FOR INDICES ---
FYERS_INDEX_SYMBOLS = {
    "NIFTY200_INDEX": "NSE:NIFTY200-INDEX",
    "INDIAVIX": "NSE:INDIAVIX-INDEX",
    "NIFTY500-INDEX": "NSE:NIFTY500-INDEX",
    "NIFTY50-INDEX": "NSE:NIFTY50-INDEX",
    "NIFTYBANK-INDEX": "NSE:NIFTYBANK-INDEX",
    "GOLD25OCTFUT": "MCX:GOLD25OCTFUT"
}

# --- GLOBAL VARIABLES FOR MULTIPROCESSING ---
global_fyers_client = None
global_interval = None
global_file_suffix = None
global_force_download = None
global_failed_symbols_list = None
global_throttled_symbols_list = None

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
    Fetches historical OHLC data with a retry mechanism and specific error handling.
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
                print(f" - Worker {current_process().pid} ERROR: Invalid symbol {symbol}. Skipping retries.")
                return INVALID_SYMBOL_ERROR
            if response.get("message") == "request limit reached":
                print(f" - Worker {current_process().pid} ERROR: API request limit reached for {symbol}. Retrying in 10 seconds...")
                time.sleep(10)
                continue
            if response.get("s") == 'ok':
                candles = response.get('candles', [])
                if not candles:
                    return pd.DataFrame()
                df = pd.DataFrame(candles, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                df['datetime'] = df['datetime'] + pd.Timedelta(hours=5, minutes=30)
                # --- FIX: Do NOT set datetime as index ---
                return df
            else:
                print(f" - Worker {current_process().pid} API Error for {symbol} (Attempt {i+1}): {response.get('message', 'Unknown error')}")
        except Exception as e:
            print(f" - Worker {current_process().pid} An exception occurred for {symbol} (Attempt {i+1}): {e}")
        if i < SCRIPT_CONFIG["api_retries"] - 1:
            time.sleep(SCRIPT_CONFIG["api_cooldown_seconds"] * (2 ** i))
    print(f" - Worker {current_process().pid} Failed to fetch data for {symbol} after {SCRIPT_CONFIG['api_retries']} attempts.")
    return pd.DataFrame()


def process_single_symbol(symbol_name):
    """
    Worker function to scrape data for a single symbol, handles incremental download by checking existing data.
    """
    global global_fyers_client, global_interval, global_file_suffix, global_force_download, global_failed_symbols_list, global_throttled_symbols_list

    if not global_fyers_client:
        print(f"Worker process {current_process().pid} has no Fyers client. Skipping {symbol_name}.")
        return

    print(f"Worker {current_process().pid} processing {symbol_name}...")

    # Determine resolution and batching parameters
    if global_interval == "daily":
        resolution = "D"
        batch_months = 6
        batch_days = None
    elif global_interval == "15min":
        resolution = "15"
        batch_months = 2
        batch_days = None
    elif global_interval == "1min":
        resolution = "1"
        batch_months = None
        batch_days = 15  # 15 days batch for 1min interval to avoid fractional months error
    else:
        resolution = "D"
        batch_months = 6
        batch_days = None

    output_path = os.path.join(SCRIPT_CONFIG["output_dir"], f"{symbol_name}_{global_file_suffix}.csv")
    to_date = datetime.now()

    existing_date_ranges = []

    # Determine ranges to download based on existing data and force flag
    if global_force_download or not os.path.exists(output_path):
        existing_date_ranges.append((pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date(), to_date.date()))
        if global_force_download:
            print(f" - Worker {current_process().pid} FORCE DOWNLOAD enabled, downloading full data from {SCRIPT_CONFIG['default_start_date']} to {to_date.date()}")
    else:
        try:
            existing_df = pd.read_csv(output_path, parse_dates=['datetime'])
            if not existing_df.empty:
                last_existing_date = existing_df['datetime'].dt.date.max()
                next_start_date = last_existing_date + timedelta(days=1)
                if next_start_date <= to_date.date():
                    existing_date_ranges.append((next_start_date, to_date.date()))
                    print(f" - Worker {current_process().pid} Incremental download for {symbol_name} from {next_start_date} to {to_date.date()}")
                else:
                    print(f" - Worker {current_process().pid} Data up to date for {symbol_name}. Skipping.")
                    return
            else:
                existing_date_ranges.append((pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date(), to_date.date()))
                print(f" - Worker {current_process().pid} Empty data file for {symbol_name}, downloading full range.")
        except Exception as e:
            existing_date_ranges.append((pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date(), to_date.date()))
            print(f" - Worker {current_process().pid} Failed to read existing data file for {symbol_name}: {e}, downloading full range.")

    if not existing_date_ranges:
        # Nothing to download
        return

    all_batches = []

    for start_date, end_date in existing_date_ranges:
        current_start = start_date
        while current_start <= end_date:
            if batch_days is not None:
                current_end = pd.to_datetime(current_start) + timedelta(days=batch_days) - timedelta(days=1)
            else:
                current_end = pd.to_datetime(current_start) + pd.DateOffset(months=batch_months) - timedelta(days=1)

            if current_end.date() > end_date:
                current_end = pd.to_datetime(end_date)

            from_str = current_start.strftime('%Y-%m-%d')
            to_str = current_end.strftime('%Y-%m-%d')

            if symbol_name in FYERS_INDEX_SYMBOLS:
                fyers_symbol = FYERS_INDEX_SYMBOLS[symbol_name]
            else:
                fyers_symbol = f"NSE:{symbol_name}-EQ"

            print(f" - Worker {current_process().pid} Fetching batch for {symbol_name}: {from_str} to {to_str}")

            batch_df = get_historical_data(global_fyers_client, fyers_symbol, resolution, from_str, to_str)

            if isinstance(batch_df, str):
                if batch_df == INVALID_SYMBOL_ERROR:
                    global_failed_symbols_list.append(symbol_name)
                    return
                elif batch_df == THROTTLED_ERROR:
                    global_throttled_symbols_list.append(symbol_name)
                    return

            if not batch_df.empty:
                all_batches.append(batch_df)

            current_start = (current_end + timedelta(days=1)).date()

    if not all_batches:
        print(f" - Worker {current_process().pid} Info: No new data returned for {symbol_name}.")
        return

    combined_df = pd.concat(all_batches).sort_values(by='datetime')

    if global_force_download or not os.path.exists(output_path):
        # --- FIX: Added index=False ---
        combined_df.to_csv(output_path, mode='w', header=True, index=False)
        print(f" - Worker {current_process().pid} Success: Saved {len(combined_df)} records for {symbol_name} to {output_path}")
    else:
        try:
            existing_df = pd.read_csv(output_path, parse_dates=['datetime'])
            merged_df = pd.concat([existing_df, combined_df]).drop_duplicates(subset='datetime').sort_values('datetime')
            # --- FIX: Added index=False ---
            merged_df.to_csv(output_path, mode='w', header=True, index=False)
            print(f" - Worker {current_process().pid} Success: Merged and saved updated {len(merged_df)} records for {symbol_name} to {output_path}")
        except Exception as e:
            # --- FIX: Added index=False ---
            combined_df.to_csv(output_path, mode='a', header=False, index=False)
            print(f" - Worker {current_process().pid} Warning: Could not merge data for {symbol_name} due to error: {e}. Appended new data only.")


def main():
    """
    Main orchestration function using multiprocessing.
    """
    parser = argparse.ArgumentParser(description="Universal Fyers Scraper for Nifty 200 Project.")

    parser.add_argument('--interval', type=str, required=False, choices=['daily', '15min', '1min'],
                        help='Specify an interval to scrape: daily, 15min, or 1min. If not provided, all intervals will be scraped.')

    parser.add_argument('--force', action='store_true', help='Force a full re-download of data, ignoring existing files.')

    parser.add_argument('--only-index', action='store_true',
                        help='Scrape data only for the indices defined in the config.')

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
    run_1min = not args.interval or args.interval == '1min'

    with Manager() as manager:
        failed_symbols = manager.list()
        throttled_symbols = manager.list()
        num_processes = SCRIPT_CONFIG["parallel_processes"]

        print(f"\nStarting parallel downloads using {num_processes} processes.")

        if run_15min:
            print(f"\n--- Starting 15-minute scrape for {len(equity_symbols)} equities... ---")
            with Pool(initializer=worker_init,
                      initargs=(access_token, "15min", "15min", args.force, failed_symbols, throttled_symbols)) as pool:
                pool.map(process_single_symbol, equity_symbols)

            print(f"\n--- Starting 15-minute scrape for {len(index_symbols)} indices... ---")
            with Pool(initializer=worker_init,
                      initargs=(access_token, "15min", "15min", args.force, failed_symbols, throttled_symbols)) as pool:
                pool.map(process_single_symbol, index_symbols)

        if run_daily:
            print(f"\n--- Starting daily scrape for {len(equity_symbols)} equities... ---")
            with Pool(initializer=worker_init,
                      initargs=(access_token, "daily", "daily", args.force, failed_symbols, throttled_symbols)) as pool:
                pool.map(process_single_symbol, equity_symbols)

            print(f"\n--- Starting daily scrape for {len(index_symbols)} indices... ---")
            with Pool(initializer=worker_init,
                      initargs=(access_token, "daily", "daily", args.force, failed_symbols, throttled_symbols)) as pool:
                pool.map(process_single_symbol, index_symbols)

        if run_1min:
            print(f"\n--- Starting 1-minute scrape for {len(equity_symbols)} equities... ---")
            with Pool(initializer=worker_init,
                      initargs=(access_token, "1min", "1min", args.force, failed_symbols, throttled_symbols)) as pool:
                pool.map(process_single_symbol, equity_symbols)

            print(f"\n--- Starting 1-minute scrape for {len(index_symbols)} indices... ---")
            with Pool(initializer=worker_init,
                      initargs=(access_token, "1min", "1min", args.force, failed_symbols, throttled_symbols)) as pool:
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
