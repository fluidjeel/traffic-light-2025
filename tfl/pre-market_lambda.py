import boto3
import os
import pandas as pd
from pytz import timezone
import datetime

# ==============================================================================
# --- LAMBDA CONFIGURATION & ENVIRONMENT VARIABLES ---
# ==============================================================================
# This Lambda function requires the following environment variables to be set:
#
# S3_BUCKET_NAME: The name of the S3 bucket where your daily data is stored.
# S3_DAILY_DATA_PREFIX: The prefix (folder) within the bucket for daily data.
# SYMBOL_LIST_KEY: The key (path) to the nifty200_fno.csv file in the S3 bucket.
# SSM_PARAM_TRADING_BIAS: The name of the SSM Parameter to update (e.g., /tfl/trading_bias).
# SNS_TOPIC_ARN_NOTIFICATIONS: The ARN of the SNS topic for sending alerts.

# --- Strategy Parameters (should match your backtester for consistency) ---
BREADTH_SMA_PERIOD = 50
TREND_FILTER_SMA_PERIOD_LONG = 30  # From the last successful long run
TREND_FILTER_SMA_PERIOD_SHORT = 100 # From the shorts backtester
BREADTH_THRESHOLD_LONG = 60.0
BREADTH_THRESHOLD_SHORT = 60.0
VIX_THRESHOLD_SHORT = 17.0

# --- Key Symbols ---
NIFTY_50_SYMBOL = 'NIFTY50-INDEX'
INDIA_VIX_SYMBOL = 'INDIAVIX'
AWS_REGION = "ap-south-1"
INDIA_TZ = timezone('Asia/Kolkata')

# ==============================================================================
# --- AWS CLIENTS (initialized globally for reuse) ---
# ==============================================================================
s3_client = boto3.client('s3')
ssm_client = boto3.client('ssm', region_name=AWS_REGION)
sns_client = boto3.client('sns', region_name=AWS_REGION)

def get_df_from_s3(bucket, key):
    """Downloads a parquet or csv file from S3 and returns a pandas DataFrame."""
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        if key.endswith('.parquet'):
            return pd.read_parquet(obj['Body'])
        elif key.endswith('.csv'):
            return pd.read_csv(obj['Body'])
        else:
            raise ValueError(f"Unsupported file type for key: {key}")
    except Exception as e:
        print(f"Error reading {key} from S3 bucket {bucket}: {e}")
        return pd.DataFrame()

def lambda_handler(event, context):
    """
    This is the main entry point for the Lambda function.
    It calculates the daily market regime and updates the SSM Parameter.
    """
    print("--- Starting Pre-Market Regime Calculation ---")
    
    # --- Load Configuration from Environment Variables ---
    bucket = os.environ['S3_BUCKET_NAME']
    data_prefix = os.environ['S3_DAILY_DATA_PREFIX']
    symbol_list_key = os.environ['SYMBOL_LIST_KEY']
    param_name = os.environ['SSM_PARAM_TRADING_BIAS']
    sns_topic = os.environ['SNS_TOPIC_ARN_NOTIFICATIONS']

    # --- 1. Fetch Necessary Data from S3 ---
    symbols_df = get_df_from_s3(bucket, symbol_list_key)
    if symbols_df.empty:
        message = "FATAL: Could not load symbol list. Setting bias to NO_TRADES."
        print(message)
        sns_client.publish(TopicArn=sns_topic, Message=message, Subject="TFL Regime Lambda ERROR")
        update_trading_bias(param_name, "NO_TRADES")
        return {'statusCode': 500, 'body': json.dumps(message)}

    symbols = symbols_df['symbol'].tolist()
    nifty_df = get_df_from_s3(bucket, f"{data_prefix}/{NIFTY_50_SYMBOL}_daily_with_indicators.parquet")
    vix_df = get_df_from_s3(bucket, f"{data_prefix}/{INDIA_VIX_SYMBOL}_daily_with_indicators.parquet")
    
    if nifty_df.empty or vix_df.empty:
        message = "FATAL: Could not load NIFTY or VIX data. Setting bias to NO_TRADES."
        print(message)
        sns_client.publish(TopicArn=sns_topic, Message=message, Subject="TFL Regime Lambda ERROR")
        update_trading_bias(param_name, "NO_TRADES")
        return {'statusCode': 500, 'body': json.dumps(message)}

    # --- 2. Calculate Market Breadth ---
    stocks_above_sma = 0
    stocks_below_sma = 0
    total_stocks_processed = 0

    for symbol in symbols:
        stock_df = get_df_from_s3(bucket, f"{data_prefix}/{symbol}_daily_with_indicators.parquet")
        if not stock_df.empty and f'sma_{BREADTH_SMA_PERIOD}' in stock_df.columns:
            last_close = stock_df['close'].iloc[-1]
            last_sma = stock_df[f'sma_{BREADTH_SMA_PERIOD}'].iloc[-1]
            if pd.notna(last_close) and pd.notna(last_sma):
                if last_close > last_sma:
                    stocks_above_sma += 1
                else:
                    stocks_below_sma += 1
                total_stocks_processed += 1
    
    breadth_pct_above = (stocks_above_sma / total_stocks_processed) * 100 if total_stocks_processed > 0 else 0
    breadth_pct_below = (stocks_below_sma / total_stocks_processed) * 100 if total_stocks_processed > 0 else 0

    # --- 3. Determine Regime Conditions ---
    last_nifty_close = nifty_df['close'].iloc[-1]
    nifty_sma_long = nifty_df[f'sma_{TREND_FILTER_SMA_PERIOD_LONG}'].iloc[-1]
    nifty_sma_short = nifty_df[f'sma_{TREND_FILTER_SMA_PERIOD_SHORT}'].iloc[-1]
    last_vix_close = vix_df['close'].iloc[-1]

    # Conditions for Longs (looser filters from last successful backtest)
    long_breadth_ok = breadth_pct_above > BREADTH_THRESHOLD_LONG
    long_trend_ok = last_nifty_close > nifty_sma_long

    # Conditions for Shorts (stricter filters)
    short_breadth_ok = breadth_pct_below > BREADTH_THRESHOLD_SHORT
    short_trend_ok = last_nifty_close < nifty_sma_short
    short_vol_ok = last_vix_close > VIX_THRESHOLD_SHORT

    # --- 4. Decide Final Trading Bias ---
    trading_bias = "NO_TRADES"
    if short_breadth_ok and short_trend_ok and short_vol_ok:
        trading_bias = "SHORTS_ONLY"
    elif long_breadth_ok and long_trend_ok:
        trading_bias = "LONGS_ONLY"

    # --- 5. Update SSM Parameter and Notify ---
    update_trading_bias(param_name, trading_bias)
    message = f"TFL Pre-Market Regime Lambda: Successfully set trading bias to {trading_bias}."
    print(message)
    sns_client.publish(TopicArn=sns_topic, Message=message, Subject="TFL Daily Trading Bias Set")
    
    return {
        'statusCode': 200,
        'body': json.dumps(f"Bias set to {trading_bias}")
    }

def update_trading_bias(param_name, bias_value):
    """Updates the specified SSM Parameter with the new bias value."""
    print(f"Updating SSM Parameter '{param_name}' to '{bias_value}'...")
    try:
        ssm_client.put_parameter(
            Name=param_name,
            Value=bias_value,
            Type='String',
            Overwrite=True
        )
        print("SSM Parameter updated successfully.")
    except Exception as e:
        print(f"FATAL: Error updating SSM Parameter: {e}")
