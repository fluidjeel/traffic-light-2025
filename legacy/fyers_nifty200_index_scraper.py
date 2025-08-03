# fyers_nifty200_index_scraper_15min.py
#
# Description:
# This script is specifically designed to download and update historical 15-MINUTE 
# data for the Nifty 200 Index. It's a necessary component for the V5 Hybrid Model,
# which requires intraday index data for its real-time market regime filters.

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
    print("Please make sure you have a config.py file with CLIENT_ID, SECRET_KEY, and REDIRECT_URI defined.")
    sys.exit()

# --- Import Fyers API Module ---
from fyers_apiv3 import fyersModel

# --- Script Configuration ---
LOG_PATH = os.getcwd()
TOKEN_FILE = "fyers_access_token.txt"
INDEX_SYMBOL = "NSE:NIFTY200-INDEX" # Fyers API symbol for Nifty 200 Index

def get_access_token():
    """
    Handles the Fyers authentication flow to get the access token.
    Reads from a file if available, otherwise generates a new one.
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
    print("2. After logging in, you will be redirected.")
    print("3. Copy the 'auth_code' from the redirected URL.")
    
    auth_code = input("4. Enter the auth_code here: ")

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
    """
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
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
                return df
            else:
                print(f"    - API Error (Attempt {i+1}/{retries}): {response.get('message', 'Unknown error')}")
                if i < retries - 1:
                    time.sleep(delay)
        except Exception as e:
            print(f"    - An exception occurred (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(delay)

    print(f"    - Failed to fetch data for {symbol} after {retries} attempts.")
    return pd.DataFrame()

if __name__ == "__main__":
    if CLIENT_ID == "YOUR_CLIENT_ID" or SECRET_KEY == "YOUR_SECRET_KEY":
        print("Please replace 'YOUR_CLIENT_ID' and 'YOUR_SECRET_KEY' in the config.py file.")
    else:
        # --- Main Execution Block ---
        
        FORCE_DOWNLOAD_FROM_DATE = None
        DEFAULT_START_DATE = "2020-01-01"
        
        output_dir = "historical_data_15min"
        os.makedirs(output_dir, exist_ok=True)
        
        # MODIFICATION: Filename is specific to the Nifty 200 Index
        output_path = os.path.join(output_dir, "NIFTY_200_15min.csv")
        
        print(f"\nProcessing Nifty 200 Index ({INDEX_SYMBOL})...")
        to_date = datetime.now()
        is_force_download = False

        if FORCE_DOWNLOAD_FROM_DATE:
            from_date = pd.to_datetime(FORCE_DOWNLOAD_FROM_DATE).date()
            print(f"  > FORCE DOWNLOAD enabled. Fetching all data from {from_date.strftime('%Y-%m-%d')}")
            is_force_download = True
        elif os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                if not existing_df.empty:
                    last_date_str = existing_df['datetime'].iloc[-1]
                    last_date = pd.to_datetime(last_date_str).date()
                    from_date = last_date
                    print(f"  > Existing data found. Fetching new data from {from_date.strftime('%Y-%m-%d')}")
                else:
                     print(f"  > Existing file is empty. Performing full download from {DEFAULT_START_DATE}")
                     from_date = pd.to_datetime(DEFAULT_START_DATE).date()
            except Exception as e:
                print(f"  > Could not read existing file {output_path}. Performing a full download. Error: {e}")
                from_date = pd.to_datetime(DEFAULT_START_DATE).date()
        else:
            print(f"  > No existing data. Performing full download from {DEFAULT_START_DATE}")
            from_date = pd.to_datetime(DEFAULT_START_DATE).date()

        if from_date > to_date.date():
             print(f"  > Data is already up to date. Skipping.")
             sys.exit()
        
        all_data_batches = []
        batch_start_date = from_date

        while batch_start_date <= to_date.date():
            batch_end_dt = pd.to_datetime(batch_start_date) + pd.DateOffset(days=90)
            
            if batch_end_dt > to_date:
                batch_end_dt = to_date

            from_date_str = batch_start_date.strftime('%Y-%m-%d')
            to_date_str = batch_end_dt.strftime('%Y-%m-%d')
            
            print(f"    > Fetching batch: {from_date_str} to {to_date_str}")

            # MODIFICATION: Resolution is "15"
            batch_df = get_historical_data(INDEX_SYMBOL, "15", from_date_str, to_date_str)
            
            if not batch_df.empty:
                all_data_batches.append(batch_df)

            time.sleep(1.1)
            batch_start_date = batch_end_dt.date() + timedelta(days=1)

        if all_data_batches:
            new_data_df = pd.concat(all_data_batches)
            new_data_df = new_data_df[~new_data_df.index.duplicated(keep='last')]
        else:
            new_data_df = pd.DataFrame()
        
        if not new_data_df.empty:
            if is_force_download or not os.path.exists(output_path):
                new_data_df.to_csv(output_path, mode='w', header=True)
                print(f"  > Success: Saved {len(new_data_df)} records to {output_path}")
            else:
                old_df = pd.read_csv(output_path, index_col='datetime', parse_dates=True)
                combined_df = pd.concat([old_df, new_data_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.sort_index(inplace=True)
                combined_df.to_csv(output_path, mode='w', header=True)
                print(f"  > Success: Updated file with {len(new_data_df)} new records at {output_path}")
        else:
            print(f"  > Info: No new data returned for the index.")
        
        print("\n--- Index data update complete! ---")
