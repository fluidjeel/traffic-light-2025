# fyers_futures_scraper.py
#
# Description:
# This script downloads historical daily data for all futures contracts of a
# given underlying (e.g., BANKNIFTY) from a specified start date to the present.
#
# Prerequisites:
# 1. A Fyers trading account and API App credentials.
# 2. A 'config.py' file with your credentials.
# 3. Required libraries installed: pip install fyers-apiv3 pandas
#
# How to use:
# 1. Fill in your credentials in 'config.py'.
# 2. Set the `UNDERLYING_SYMBOL` and `START_DATE_FOR_FUTURES` variables below.
# 3. Run the script. It will create a 'futures_data' folder and save the data
#    for each historical contract it finds.

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import time

# --- Import Configuration ---
try:
    from config import CLIENT_ID, SECRET_KEY, REDIRECT_URI
except ImportError:
    print("Error: Could not import from config.py.")
    sys.exit()

# --- Import Fyers API Module ---
from fyers_apiv3 import fyersModel

# --- Script Configuration ---
LOG_PATH = os.getcwd()
TOKEN_FILE = "fyers_access_token.txt"

def get_access_token():
    """
    Handles the Fyers authentication flow to get the access token.
    """
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return f.read().strip()

    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type='code',
        grant_type='authorization_code'
    )
    auth_url = session.generate_authcode()
    print("--- Fyers Login ---")
    print(f"1. Go to this URL and log in: {auth_url}")
    auth_code = input("2. Enter the auth_code from the redirected URL: ")
    session.set_token(auth_code)
    response = session.generate_token()

    if response.get("access_token"):
        access_token = response["access_token"]
        with open(TOKEN_FILE, 'w') as f:
            f.write(access_token)
        print("Access token generated and saved successfully.")
        return access_token
    else:
        print(f"Failed to generate access token: {response}")
        return None

def generate_futures_symbol_for_date(underlying, target_date):
    """
    Generates the correct Fyers API symbol for a futures contract for a given month and year.
    :param underlying: The base symbol, e.g., "BANKNIFTY".
    :param target_date: A datetime object for the desired contract month.
    """
    year_short = target_date.strftime('%y')
    month_abbr = target_date.strftime("%b").upper()
    
    # Fyers futures symbol format: NSE:BANKNIFTY25JULFUT
    return f"NSE:{underlying}{year_short}{month_abbr}FUT"

def get_historical_data(fyers, symbol, resolution, from_date, to_date):
    """
    Fetches historical OHLC data for a given security.
    """
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": from_date,
        "range_to": to_date,
        "cont_flag": "1"
    }
    try:
        response = fyers.history(data=data)
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
            # This is a normal occurrence if a contract for a specific month doesn't exist.
            # We print it for info but don't treat it as a critical error.
            print(f"    - Info: {response.get('message', 'Unknown error')} for symbol {symbol}")
            return pd.DataFrame()
    except Exception as e:
        print(f"    - An exception occurred: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # --- FUTURES CONFIGURATION ---
    UNDERLYING_SYMBOL = "BANKNIFTY"
    START_DATE_FOR_FUTURES = "2023-01-01" # Set the date from which you want to start fetching contracts

    # --- Initialize Fyers Client ---
    access_token = get_access_token()
    if not access_token:
        sys.exit()

    fyers = fyersModel.FyersModel(
        client_id=CLIENT_ID,
        is_async=False,
        token=access_token,
        log_path=LOG_PATH
    )

    # --- Main Execution ---
    output_dir = "futures_data"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Loop through each month from the start date to today
    start_date = datetime.strptime(START_DATE_FOR_FUTURES, '%Y-%m-%d')
    end_date = datetime.now()
    current_month_start = datetime(start_date.year, start_date.month, 1)

    while current_month_start <= end_date:
        # 2. Generate the futures symbol for the current month in the loop
        futures_symbol = generate_futures_symbol_for_date(UNDERLYING_SYMBOL, current_month_start)
        print(f"\nAttempting to fetch data for contract: {futures_symbol}")

        # 3. Define date range for the specific contract (e.g., 90 days)
        to_date_str = (current_month_start + pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        from_date_str = (current_month_start - pd.DateOffset(days=5)).strftime('%Y-%m-%d')

        # 4. Fetch the data
        print(f"  > Searching for data in range: {from_date_str} to {to_date_str}...")
        historical_df = get_historical_data(fyers, futures_symbol, "D", from_date_str, to_date_str)

        # 5. Save the data if found
        if not historical_df.empty:
            safe_symbol_name = futures_symbol.replace(":", "_")
            output_path = os.path.join(output_dir, f"{safe_symbol_name}_daily.csv")
            historical_df.to_csv(output_path)
            print(f"  > Success: Found and saved {len(historical_df)} records to {output_path}")
        else:
            print(f"  > No data found for this contract.")
        
        # Move to the next month
        current_month_start += pd.DateOffset(months=1)
        
        # Add a small delay to respect API rate limits
        time.sleep(1.1)

    print("\n--- All futures contracts download complete! ---")
