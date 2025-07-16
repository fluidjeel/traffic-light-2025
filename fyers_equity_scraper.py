# fyers_equity_scraper.py
#
# Description:
# This script intelligently downloads and updates historical daily data for all
# stocks in the Nifty 200 index. It only fetches missing data since the last run.
#
# Prerequisites:
# 1. A Fyers trading account and API App credentials.
# 2. A 'config.py' file with your credentials.
# 3. A 'nifty200.csv' file with the stock symbols.
# 4. Required libraries installed: pip install fyers-apiv3 pandas
#
# How to use:
# 1. Fill in your credentials in 'config.py'.
# 2. Make sure 'nifty200.csv' is in the same directory.
# 3. Run the script.
#    - On the first run, it downloads the full history for each stock.
#    - On subsequent runs, it only fetches and appends the new, missing data.

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import time

# --- Import Configuration ---
# Make sure you have a config.py file in the same directory
try:
    from config import CLIENT_ID, SECRET_KEY, REDIRECT_URI
except ImportError:
    print("Error: Could not import from config.py.")
    print("Please make sure you have a config.py file with CLIENT_ID, SECRET_KEY, and REDIRECT_URI defined.")
    sys.exit()


# --- Import Fyers API Module ---
from fyers_apiv3 import fyersModel

# --- Script Configuration ---
LOG_PATH = os.getcwd() # Path to store logs
TOKEN_FILE = "fyers_access_token.txt"

def get_access_token():
    """
    Handles the Fyers authentication flow to get the access token.
    Reads from a file if available, otherwise generates a new one.
    """
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return f.read().strip()

    # Step 1: Generate the auth code generation URL
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
    print("2. After logging in, you will be redirected.")
    print("3. Copy the 'auth_code' from the redirected URL.")
    
    auth_code = input("4. Enter the auth_code here: ")

    # Step 2: Generate the access token from the auth_code
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

# --- Fyers API Client Initialization ---
access_token = get_access_token()

if not access_token:
    print("Could not obtain access token. Exiting.")
    exit()

try:
    fyers = fyersModel.FyersModel(
        client_id=CLIENT_ID,
        is_async=False,
        token=access_token,
        log_path=LOG_PATH
    )
    print("Fyers API client initialized successfully.")
    profile = fyers.get_profile()
    if profile['s'] != 'ok':
         print("Profile fetch failed. The access token might be invalid. Please delete fyers_access_token.txt and run again.")
         exit()
    print(f"Welcome, {profile['data']['name']}!")
except Exception as e:
    print(f"Error initializing Fyers API client: {e}")
    exit()

def get_historical_data(symbol, resolution, from_date, to_date, retries=3, delay=2):
    """
    Fetches historical OHLC data for a given security with a retry mechanism.
    :param symbol: e.g., "NSE:RELIANCE-EQ"
    :param resolution: "D" for daily, "1" for 1-minute, etc.
    :param from_date: "YYYY-MM-DD"
    :param to_date: "YYYY-MM-DD"
    :param retries: Number of times to retry on failure.
    :param delay: Seconds to wait between retries.
    """
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1", # 0 for epoch, 1 for YYYY-MM-DD
        "range_from": from_date,
        "range_to": to_date,
        "cont_flag": "1"
    }
    
    for i in range(retries):
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
                return df # Success, exit the function
            else:
                print(f"    - API Error (Attempt {i+1}/{retries}): {response.get('message', 'Unknown error')}")
                if i < retries - 1:
                    print(f"    - Retrying in {delay} seconds...")
                    time.sleep(delay)
        except Exception as e:
            print(f"    - An exception occurred (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                print(f"    - Retrying in {delay} seconds...")
                time.sleep(delay)

    # If all retries fail
    print(f"    - Failed to fetch data for {symbol} after {retries} attempts.")
    return pd.DataFrame()

if __name__ == "__main__":
    if CLIENT_ID == "YOUR_CLIENT_ID" or SECRET_KEY == "YOUR_SECRET_KEY":
        print("Please replace 'YOUR_CLIENT_ID' and 'YOUR_SECRET_KEY' in the config.py file.")
    else:
        # --- Main Execution Block for Incremental Download ---
        
        DEFAULT_START_DATE = "2000-01-01"
        
        # 1. Define the output directory
        output_dir = "historical_data"
        os.makedirs(output_dir, exist_ok=True)

        # 2. Read the list of stocks
        try:
            stock_list_df = pd.read_csv("nifty200.csv")
            symbols = stock_list_df["Symbol"].tolist()
        except FileNotFoundError:
            print("Error: 'nifty200.csv' not found. Please make sure the file is in the same directory.")
            sys.exit()

        total_stocks = len(symbols)
        print(f"\nFound {total_stocks} stocks to process.")

        # 3. Loop through each stock and fetch data
        for i, symbol_name in enumerate(symbols):
            print(f"\nProcessing {symbol_name} ({i+1}/{total_stocks})...")
            
            output_path = os.path.join(output_dir, f"{symbol_name}_daily.csv")
            to_date = datetime.now()

            # --- INTELLIGENT DATE LOGIC ---
            if os.path.exists(output_path):
                # File exists, find the last date and fetch from the next day
                try:
                    existing_df = pd.read_csv(output_path)
                    last_date_str = existing_df['datetime'].iloc[-1]
                    last_date = pd.to_datetime(last_date_str).date()
                    from_date = last_date + timedelta(days=1)
                    print(f"  > Existing data found. Fetching new data from {from_date.strftime('%Y-%m-%d')}")
                except Exception as e:
                    print(f"  > Could not read existing file {output_path}. Performing a full download. Error: {e}")
                    from_date = pd.to_datetime(DEFAULT_START_DATE).date()
            else:
                # File does not exist, perform a full download
                print(f"  > No existing data. Performing full download from {DEFAULT_START_DATE}")
                from_date = pd.to_datetime(DEFAULT_START_DATE).date()

            # Ensure from_date is not in the future
            if from_date > to_date.date():
                 print(f"  > Data is already up to date. Skipping.")
                 continue

            # Format dates for the API call
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')
            
            # Format symbol for Fyers API
            fyers_symbol = f"NSE:{symbol_name}-EQ"
            
            # Fetch the data
            new_data_df = get_historical_data(fyers_symbol, "D", from_date_str, to_date_str)
            
            # Save or append the data
            if not new_data_df.empty:
                if os.path.exists(output_path):
                    # Append new data without headers
                    new_data_df.to_csv(output_path, mode='a', header=False)
                    print(f"  > Success: Appended {len(new_data_df)} new records to {output_path}")
                else:
                    # Save new file with headers
                    new_data_df.to_csv(output_path)
                    print(f"  > Success: Saved {len(new_data_df)} records to {output_path}")
            else:
                print(f"  > Info: No new data returned for {symbol_name}.")

            # Add a small delay to respect API rate limits
            time.sleep(1.1)
            
        print("\n--- Data update complete! ---")
