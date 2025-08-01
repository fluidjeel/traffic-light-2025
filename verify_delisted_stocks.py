import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import time

# --- Import Configuration from your existing files ---
try:
    from config import CLIENT_ID, SECRET_KEY, REDIRECT_URI
except ImportError:
    print("Error: Could not import from config.py.")
    print("Please make sure you have a config.py file with CLIENT_ID, SECRET_KEY, and REDIRECT_URI defined.")
    sys.exit()

# --- Import Fyers API Module ---
from fyers_apiv3 import fyersModel

# --- Script Configuration ---
DELISTED_STOCKS_CSV = "delisted_stocks.csv"
VERIFIED_STOCKS_CSV = "delisted_stocks_verified.csv"
TOKEN_FILE = "fyers_access_token.txt"
LOG_PATH = os.getcwd()
MIN_YEARS_DATA = 3  # Minimum years of data required
BATCH_DAYS = 90     # Same as your existing scrapers

def get_access_token():
    """Identical to your existing scrapers' authentication"""
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
        with open(TOKEN_FILE, 'w') as f:
            f.write(response["access_token"])
        return response["access_token"]
    else:
        print(f"Failed to generate access token: {response}")
        return None

def init_fyers_client():
    """Initialize FYERS client with error handling"""
    access_token = get_access_token()
    if not access_token:
        return None

    try:
        fyers = fyersModel.FyersModel(
            client_id=CLIENT_ID,
            is_async=False,
            token=access_token,
            log_path=LOG_PATH
        )
        # Verify connection
        profile = fyers.get_profile()
        if profile['s'] == 'ok':
            return fyers
    except Exception as e:
        print(f"Error initializing FYERS client: {e}")
    return None

def get_historical_data(fyers, symbol, from_date, to_date):
    """Modified version of your existing data fetcher"""
    data = {
        "symbol": f"NSE:{symbol}-EQ",
        "resolution": "1D",
        "date_format": "1",
        "range_from": from_date.strftime('%Y-%m-%d'),
        "range_to": to_date.strftime('%Y-%m-%d'),
        "cont_flag": "1"
    }
    
    for attempt in range(3):
        try:
            response = fyers.history(data=data)
            if response.get('s') == 'ok':
                candles = response.get('candles', [])
                if candles:
                    return len(candles)
            time.sleep(1.1)  # Respect rate limits
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {symbol}: {str(e)}")
            time.sleep(2)
    
    return 0

def check_stock_data(fyers, symbol):
    """Check data availability with batch fetching like your scrapers"""
    to_date = datetime.now()
    from_date = to_date - timedelta(days=365*MIN_YEARS_DATA)
    total_candles = 0
    
    batch_start = from_date
    while batch_start <= to_date:
        batch_end = min(batch_start + timedelta(days=BATCH_DAYS), to_date)
        
        candles = get_historical_data(fyers, symbol, batch_start, batch_end)
        if candles == 0:
            return 0  # Early exit if any batch fails
        
        total_candles += candles
        batch_start = batch_end + timedelta(days=1)
    
    return total_candles

def main():
    # Initialize FYERS client
    fyers = init_fyers_client()
    if not fyers:
        print("Failed to initialize FYERS client. Check auth tokens.")
        return

    # Load delisted stocks
    try:
        df = pd.read_csv(DELISTED_STOCKS_CSV)
        print(f"Loaded {len(df)} stocks from {DELISTED_STOCKS_CSV}")
    except Exception as e:
        print(f"Error loading {DELISTED_STOCKS_CSV}: {e}")
        return

    # Process each stock
    results = []
    for idx, row in df.iterrows():
        symbol = row['Symbol']
        print(f"Checking {symbol} ({idx+1}/{len(df)})...", end=" ", flush=True)
        
        candles = check_stock_data(fyers, symbol)
        available = candles > 0
        years = round(candles/252, 1) if available else 0
        
        results.append({
            "Symbol": symbol,
            "Company": row.get("Company", ""),
            "Data_Available": available,
            "Candles_Found": candles,
            "Years_Covered": years,
            "Last_Checked": datetime.now().strftime("%Y-%m-%d")
        })
        
        print(f"{'✅' if available else '❌'} {candles} candles ({years} years)")

    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(VERIFIED_STOCKS_CSV, index=False)
    print(f"\nVerification complete. Results saved to {VERIFIED_STOCKS_CSV}")

    # Summary
    available = result_df[result_df["Data_Available"]]
    print(f"\nStocks with sufficient data ({MIN_YEARS_DATA}+ years): {len(available)}")
    if len(available) > 0:
        print(available[["Symbol", "Company", "Years_Covered"]])

if __name__ == "__main__":
    main()