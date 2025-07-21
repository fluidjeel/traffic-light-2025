import json
import os
import boto3
import pandas as pd
from datetime import datetime, timedelta
import time
from io import StringIO, BytesIO

# This script is designed to be run as an AWS Lambda function.
# It requires a deployment package containing:
# - this script
# - fyers_apiv3 library
# - pandas, numpy libraries

# --- Fyers API Client (to be initialized later) ---
from fyers_apiv3 import fyersModel
fyers = None

# --- Environment Variables (must be set in Lambda configuration) ---
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
TOKEN_SECRET_NAME = os.environ.get('TOKEN_SECRET_NAME')
CLIENT_ID = os.environ.get('FYERS_CLIENT_ID')
NIFTY_LIST_KEY = os.environ.get('NIFTY_LIST_KEY') # e.g., "nifty200.csv"

# --- S3 Client ---
s3_client = boto3.client('s3')

def get_access_token_from_secrets():
    """Retrieves the access token from AWS Secrets Manager."""
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=TOKEN_SECRET_NAME)
    secret = json.loads(response['SecretString'])
    return secret.get('access_token')

def get_historical_data(symbol, from_date, to_date):
    """Fetches historical data from Fyers API."""
    global fyers
    data = {"symbol": symbol, "resolution": "D", "date_format": "1", "range_from": from_date, "range_to": to_date, "cont_flag": "1"}
    response = fyers.history(data=data)
    if response.get("s") == 'ok':
        candles = response.get('candles', [])
        if not candles: return pd.DataFrame()
        df = pd.DataFrame(candles, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s') + pd.Timedelta(hours=5, minutes=30)
        df.set_index('datetime', inplace=True)
        return df
    else:
        print(f"    - API Error for {symbol}: {response.get('message', 'Unknown error')}")
        return pd.DataFrame()

def calculate_atr(df, period=14):
    """Calculates ATR."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def lambda_handler(event, context):
    """
    Main Lambda handler for the daily data pipeline.
    """
    global fyers
    print("Data Pipeline Lambda started.")
    
    try:
        # --- Initialization ---
        access_token = get_access_token_from_secrets()
        if not access_token: raise Exception("Could not retrieve access token from Secrets Manager.")
        
        fyers = fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=access_token)
        print("Fyers API client initialized.")

        # --- 1. Data Acquisition (Scraping) ---
        print("Starting data acquisition...")
        
        # Get symbol list from S3
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=NIFTY_LIST_KEY)
        symbols_df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        symbols = symbols_df['Symbol'].tolist()
        symbols.append('NIFTY200_INDEX') # Also update the index

        to_date = datetime.now().strftime('%Y-%m-%d')

        for symbol in symbols:
            is_index = symbol.endswith('_INDEX')
            fyers_symbol = f"NSE:{symbol.replace('_INDEX', '')}-{'INDEX' if is_index else 'EQ'}"
            raw_data_key = f"historical_data/{symbol}{'' if is_index else '_EQ'}_daily.csv"
            
            from_date = '2020-01-01'
            try:
                # Check for existing data to perform incremental update
                head_obj = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=raw_data_key)
                existing_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=raw_data_key)
                existing_df = pd.read_csv(BytesIO(existing_obj['Body'].read()))
                last_date = pd.to_datetime(existing_df['datetime'].iloc[-1]).date()
                from_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            except s3_client.exceptions.NoSuchKey:
                print(f"No existing data for {symbol}, performing full download.")
            
            if pd.to_datetime(from_date).date() > datetime.now().date():
                print(f"{symbol} is already up-to-date. Skipping download.")
                continue

            print(f"  > Fetching {symbol} from {from_date} to {to_date}")
            new_data_df = get_historical_data(fyers_symbol, from_date, to_date)
            time.sleep(1) # API rate limiting

            if not new_data_df.empty:
                # Append or write new data to S3
                try:
                    # If file existed, append
                    csv_buffer = StringIO()
                    new_data_df.to_csv(csv_buffer)
                    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=raw_data_key, Body=existing_obj['Body'].read().decode('utf-8') + csv_buffer.getvalue().split('\n', 1)[1])
                except NameError: # existing_obj not defined
                    # If file is new, write with header
                    csv_buffer = StringIO()
                    new_data_df.to_csv(csv_buffer)
                    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=raw_data_key, Body=csv_buffer.getvalue())
        
        print("Data acquisition complete.")

        # --- 2. Indicator Calculation ---
        print("Starting indicator calculation...")
        timeframes = {'daily': 'D', '2day': '2D', 'weekly': 'W-MON', 'monthly': 'MS'}
        agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

        for symbol in symbols:
            is_index = symbol.endswith('_INDEX')
            raw_data_key = f"historical_data/{symbol}{'' if is_index else '_EQ'}_daily.csv"
            
            try:
                obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=raw_data_key)
                df_daily = pd.read_csv(BytesIO(obj['Body'].read()), index_col=0, parse_dates=True)
                df_daily.columns = [col.lower() for col in df_daily.columns]

                for tf_name, tf_rule in timeframes.items():
                    df = df_daily.copy() if tf_name == 'daily' else df_daily.resample(tf_rule).agg(agg_rules)
                    df.dropna(inplace=True)
                    if df.empty: continue

                    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
                    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
                    df['atr_14'] = calculate_atr(df, period=14)
                    df['atr_ma_30'] = df['atr_14'].rolling(window=30).mean()

                    dest_key = f"data/processed/{tf_name}/{symbol.replace('_INDEX', '')}{'_INDEX' if is_index else ''}_daily_with_indicators.csv"
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer)
                    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=dest_key, Body=csv_buffer.getvalue())
            except Exception as e:
                print(f"Error processing indicators for {symbol}: {e}")
        
        print("Indicator calculation complete.")
        return {'statusCode': 200, 'body': json.dumps('Data pipeline executed successfully!')}

    except Exception as e:
        print(f"An error occurred in the data pipeline: {str(e)}")
        # Optionally, send an SNS failure notification here
        return {'statusCode': 500, 'body': json.dumps(f'Error: {str(e)}')}

