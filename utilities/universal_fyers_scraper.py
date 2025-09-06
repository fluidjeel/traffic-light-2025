# universal_fyers_scraper.py (Enhanced for real-time error logging and detailed progress bar)
#
# Description:
# This scraper downloads historical OHLC data from Fyers API for 15-min and daily timeframes.
#
# NEW FEATURES:
# - Detailed Progress Bar: The progress bar now shows the symbol currently being processed.
# - Real-time Error Logging: Errors and warnings are printed to the console as they happen
#   without disrupting the tqdm progress bar.
# - Graceful Exit: The script properly handles Ctrl+C (KeyboardInterrupt).
# - Automatic Retry on Throttle: Symbols that fail due to API rate limits are collected and automatically retried.

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import time
from multiprocessing import Pool, Manager
import signal
from tqdm import tqdm

# --- Project Root Configuration for Portability ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

try:
    import config
except ImportError:
    print("FATAL ERROR: config.py NOT FOUND in the project root.", file=sys.stderr)
    sys.exit(1)

try:
    from fyers_apiv3 import fyersModel
except ImportError:
    print("FATAL ERROR: fyers_apiv3 library NOT FOUND. Please install it using: pip install fyers-apiv3 tqdm", file=sys.stderr)
    sys.exit(1)

# --- SCRIPT-SPECIFIC SETTINGS ---
SCRIPT_CONFIG = {
    "output_dir": os.path.join(PROJECT_ROOT, "data", "universal_historical_data"),
    "nifty_list_csv": os.path.join(PROJECT_ROOT, "nifty500.csv"),
    "index_list": ["NIFTY200_INDEX", "INDIAVIX", "NIFTY500-INDEX", "NIFTY50-INDEX", "NIFTYBANK-INDEX", "GOLD25OCTFUT"],
    "default_start_date": "2018-01-01",
    "token_file": os.path.join(PROJECT_ROOT, "fyers_access_token.txt"),
    "log_path": PROJECT_ROOT,
    "api_cooldown_seconds": 1.1,
    "api_retries": 3,
    "parallel_processes": 4
}

# --- SYMBOL MAPPING ---
FYERS_INDEX_SYMBOLS = {
    "NIFTY200_INDEX": "NSE:NIFTY200-INDEX", "INDIAVIX": "NSE:INDIAVIX-INDEX", "NIFTY500-INDEX": "NSE:NIFTY500-INDEX",
    "NIFTY50-INDEX": "NSE:NIFTY50-INDEX", "NIFTYBANK-INDEX": "NSE:NIFTYBANK-INDEX", "GOLD25OCTFUT": "MCX:GOLD25OCTFUT"
}

# --- GLOBAL VARIABLES ---
global_fyers_client = None
global_interval = None
global_file_suffix = None
global_force_download = None

def worker_init(access_token, interval, file_suffix, force_download):
    """Initializes the Fyers client and settings for each worker process."""
    global global_fyers_client, global_interval, global_file_suffix, global_force_download
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        global_fyers_client = fyersModel.FyersModel(
            client_id=config.CLIENT_ID, is_async=False, token=access_token, log_path=SCRIPT_CONFIG["log_path"]
        )
        global_interval = interval
        global_file_suffix = file_suffix
        global_force_download = force_download
    except Exception as e:
        # This error is critical and will be printed directly
        print(f"WORKER INIT FAILED: {e}", file=sys.stderr)
        global_fyers_client = None

def get_access_token():
    """Handles Fyers authentication to get the access token."""
    token_path = SCRIPT_CONFIG["token_file"]
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            return f.read().strip()

    session = fyersModel.SessionModel(
        client_id=config.CLIENT_ID, secret_key=config.SECRET_KEY, redirect_uri=config.REDIRECT_URI,
        response_type='code', grant_type='authorization_code'
    )
    auth_url = session.generate_authcode()
    print(f"--- Fyers Login Required ---\n1. Go to this URL and log in: {auth_url}\n2. Copy the 'auth_code' from the redirected URL.")
    auth_code = input("3. Enter the auth_code here: ")
    session.set_token(auth_code)
    response = session.generate_token()
    
    if access_token := response.get("access_token"):
        with open(token_path, 'w') as f: f.write(access_token)
        print("Access token saved successfully.")
        return access_token
    else:
        print(f"Failed to generate access token: {response}", file=sys.stderr)
        return None

def get_historical_data(symbol, resolution, from_date, to_date):
    """Fetches historical data with retry logic and returns a status tuple."""
    data = {"symbol": symbol, "resolution": resolution, "date_format": "1", "range_from": from_date, "range_to": to_date, "cont_flag": "1"}
    for i in range(SCRIPT_CONFIG["api_retries"]):
        try:
            response = global_fyers_client.history(data=data)
            message = response.get("message", "")
            if message == "Invalid symbol provided": return "INVALID", f"Invalid symbol: {symbol}", None
            if message == "request limit reached": return "THROTTLED", f"API rate limit reached for {symbol}", None
            
            if response.get("s") == 'ok':
                candles = response.get('candles', [])
                if not candles: return "SUCCESS", "No new data", pd.DataFrame()
                df = pd.DataFrame(candles, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s') + pd.Timedelta(hours=5, minutes=30)
                return "SUCCESS", f"Fetched {len(df)} records", df
        except Exception as e:
            if i == SCRIPT_CONFIG["api_retries"] - 1:
                return "FAIL", f"Exception after retries for {symbol}: {e}", None
        time.sleep(SCRIPT_CONFIG["api_cooldown_seconds"] * (2 ** i))
    return "FAIL", f"All retries failed for {symbol}", None

def process_single_symbol(symbol_name):
    """Worker function to scrape data for a single symbol."""
    if not global_fyers_client: return symbol_name, "FAIL", "Worker not initialized"

    resolution, batch_months = ("D", 6) if global_interval == "daily" else ("15", 2)
    output_path = os.path.join(SCRIPT_CONFIG["output_dir"], f"{symbol_name}_{global_file_suffix}.csv")
    to_date = datetime.now()
    ranges_to_download = []

    if global_force_download or not os.path.exists(output_path):
        ranges_to_download.append((pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date(), to_date.date()))
    else:
        try:
            existing_df = pd.read_csv(output_path, parse_dates=['datetime'])
            if not existing_df.empty:
                if (next_start := existing_df['datetime'].dt.date.max() + timedelta(days=1)) <= to_date.date():
                    ranges_to_download.append((next_start, to_date.date()))
                else:
                    return symbol_name, "SUCCESS", "Data up to date"
            else:
                ranges_to_download.append((pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date(), to_date.date()))
        except Exception:
            ranges_to_download.append((pd.to_datetime(SCRIPT_CONFIG["default_start_date"]).date(), to_date.date()))
    
    if not ranges_to_download: return symbol_name, "SUCCESS", "No date range to download"

    all_batches = []
    for start_date, end_date in ranges_to_download:
        current_start = start_date
        while current_start <= end_date:
            current_end = min(pd.to_datetime(end_date), pd.to_datetime(current_start) + pd.DateOffset(months=batch_months) - timedelta(days=1))
            from_str, to_str = current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d')
            fyers_symbol = FYERS_INDEX_SYMBOLS.get(symbol_name, f"NSE:{symbol_name}-EQ")
            
            status, msg, batch_df = get_historical_data(fyers_symbol, resolution, from_str, to_str)

            if status != "SUCCESS": return symbol_name, status, msg
            if batch_df is not None and not batch_df.empty: all_batches.append(batch_df)
            
            current_start = (current_end + timedelta(days=1)).date()

    if not all_batches: return symbol_name, "SUCCESS", "No new data found in range"
    
    combined_df = pd.concat(all_batches).sort_values(by='datetime')
    try:
        if not global_force_download and os.path.exists(output_path):
            existing_df = pd.read_csv(output_path, parse_dates=['datetime'])
            combined_df = pd.concat([existing_df, combined_df]).drop_duplicates(subset='datetime').sort_values('datetime')
        
        combined_df.to_csv(output_path, index=False)
        return symbol_name, "SUCCESS", f"Saved {len(combined_df)} total records"
    except Exception as e:
        return symbol_name, "FAIL", f"Could not save file: {e}"

def run_scrape_pass(symbols, interval, access_token, force, failed_list, throttled_list):
    """Helper function to run a scraping pass with a progress bar and real-time error logging."""
    if not symbols: return
    
    file_suffix = "15min" if interval == "15min" else "daily"
    init_args = (access_token, interval, file_suffix, force)
    
    desc = f"Scraping {interval} data"
    # --- ENHANCEMENT: Use a context manager for tqdm to get a handle for updating the postfix ---
    with tqdm(total=len(symbols), desc=desc, dynamic_ncols=True) as pbar:
        with Pool(processes=SCRIPT_CONFIG["parallel_processes"], initializer=worker_init, initargs=init_args) as pool:
            for symbol, status, message in pool.imap_unordered(process_single_symbol, symbols):
                # --- ENHANCEMENT: Update the progress bar with the current symbol ---
                pbar.set_postfix_str(f"Processing: {symbol}", refresh=True)
                
                if status == "FAIL" or status == "INVALID":
                    tqdm.write(f"ERROR ({symbol}): {message}")
                    failed_list.append(symbol)
                elif status == "THROTTLED":
                    tqdm.write(f"WARNING ({symbol}): {message}")
                    throttled_list.append(symbol)
                
                pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Universal Fyers Scraper.")
    parser.add_argument('--interval', type=str, choices=['daily', '15min'], help='Scrape a specific interval.')
    parser.add_argument('--force', action='store_true', help='Force a full re-download.')
    parser.add_argument('--only-index', action='store_true', help='Scrape only indices.')
    args = parser.parse_args()

    access_token = get_access_token()
    if not access_token: sys.exit(1)

    os.makedirs(SCRIPT_CONFIG["output_dir"], exist_ok=True)
    
    equity_symbols = []
    if not args.only_index:
        try:
            equity_symbols = pd.read_csv(SCRIPT_CONFIG["nifty_list_csv"])["Symbol"].tolist()
        except Exception as e:
            print(f"WARNING: Could not load equity list: {e}")
    
    all_symbols = sorted(list(set(equity_symbols + SCRIPT_CONFIG["index_list"])))

    with Manager() as manager:
        failed, throttled = manager.list(), manager.list()
        try:
            if not args.interval or args.interval == '15min':
                run_scrape_pass(all_symbols, "15min", access_token, args.force, failed, throttled)
            if not args.interval or args.interval == 'daily':
                run_scrape_pass(all_symbols, "daily", access_token, args.force, failed, throttled)

            if throttled:
                symbols_to_retry = sorted(list(set(throttled)))
                throttled[:] = [] # Clear list for the retry pass
                tqdm.write("\n" + "="*60 + f"\n--- Re-attempting {len(symbols_to_retry)} rate-limited symbols... ---\n" + "="*60)
                time.sleep(5)

                if not args.interval or args.interval == '15min':
                    run_scrape_pass(symbols_to_retry, "15min", access_token, args.force, failed, throttled)
                if not args.interval or args.interval == 'daily':
                    run_scrape_pass(symbols_to_retry, "daily", access_token, args.force, failed, throttled)
        except KeyboardInterrupt:
            print("\n\n!!! KeyboardInterrupt detected. Terminating... !!!", file=sys.stderr)
            sys.exit(1)

        print("\n--- Universal scraping process complete! ---")
        if failed:
            print("\n!!! The following symbols failed (invalid or persistent API errors): !!!")
            for symbol in sorted(list(set(failed))): print(f"- {symbol}")
        if throttled:
            print("\n!!! The following symbols were SKIPPED due to rate-limiting (even after retry): !!!")
            for symbol in sorted(list(set(throttled))): print(f"- {symbol}")

if __name__ == "__main__":
    main()

