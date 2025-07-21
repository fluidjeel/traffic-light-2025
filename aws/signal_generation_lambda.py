import json
import os
import boto3
import pandas as pd
from io import StringIO, BytesIO
import math
from datetime import datetime

# This script is designed to be run as an AWS Lambda function.
# It reuses the core logic from your final_backtester.py script.

# --- Environment Variables ---
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
NIFTY_LIST_KEY = os.environ.get('NIFTY_LIST_KEY') # e.g., "nifty200.csv"
TRADE_PLAN_KEY = os.environ.get('TRADE_PLAN_KEY') # e.g., "trade_plan.json"

# --- S3 Client ---
s3_client = boto3.client('s3')

# --- Re-using your backtester's helper functions ---
def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 2 
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def lambda_handler(event, context):
    """
    Main Lambda handler for generating the daily trade plan.
    """
    print("Signal Generation Lambda started.")
    
    try:
        # --- CONFIGURATION (can be passed in the event or set as env vars) ---
        cfg = {
            'risk_per_trade_percent': 4.0,
            'timeframe': 'daily', 
            'ema_period': 30,
            'stop_loss_lookback': 5,
            'market_regime_filter': True,
            'regime_index_symbol': 'NIFTY200',
            'regime_ma_period': 50,
            'volume_filter': True,
            'volume_ma_period': 20,
            'volume_multiplier': 1.3,
            'rs_filter': True,
            'rs_index_symbol': 'NIFTY200',
            'rs_period': 30, 
            'rs_outperformance_pct': 0.0
        }
        
        data_folder = f"data/processed/{cfg['timeframe']}"
        
        # --- Load Data from S3 ---
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=NIFTY_LIST_KEY)
        symbols = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))['Symbol'].tolist()

        index_key = f"data/processed/daily/{cfg['regime_index_symbol']}_INDEX_daily_with_indicators.csv"
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=index_key)
        index_df_daily = pd.read_csv(BytesIO(obj['Body'].read()), index_col=0, parse_dates=True)
        index_df_daily.columns = [col.lower() for col in index_df_daily.columns]
        if cfg['rs_filter']:
            index_df_daily['return'] = index_df_daily['close'].pct_change(periods=cfg['rs_period']) * 100

        stock_data = {}
        for symbol in symbols:
            key = f"{data_folder}/{symbol}_daily_with_indicators.csv"
            try:
                obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
                df = pd.read_csv(BytesIO(obj['Body'].read()), index_col=0, parse_dates=True)
                df.columns = [col.lower() for col in df.columns]
                df['red_candle'] = df['close'] < df['open']
                df['green_candle'] = df['close'] > df['open']
                if cfg['volume_filter']: df['volume_ma'] = df['volume'].rolling(window=cfg['volume_ma_period']).mean()
                if cfg['rs_filter']: df['return'] = df['close'].pct_change(periods=cfg['rs_period']) * 100
                stock_data[symbol] = df
            except s3_client.exceptions.NoSuchKey:
                continue
        
        print(f"Loaded data for {len(stock_data)} symbols.")

        # --- Generate Trade Plan ---
        trade_plan = {'new_entries': [], 'exits': [], 'modifications': []}
        date = index_df_daily.index[-1] # Always run for the latest available day
        
        # In a live system, you would get current positions and cash from the broker API
        # For this example, we assume a static portfolio state for signal generation
        portfolio_equity = 1000000 
        available_cash = 1000000
        open_positions = [] # This would be fetched from the broker

        market_uptrend = True
        if cfg['market_regime_filter']:
            if index_df_daily.loc[date]['close'] < index_df_daily.loc[date][f"ema_{cfg['regime_ma_period']}"]: market_uptrend = False

        if market_uptrend:
            for symbol, df in stock_data.items():
                if symbol in open_positions: continue
                if date not in df.index: continue
                try:
                    loc = df.index.get_loc(date)
                    if loc < max(cfg['rs_period'], 2): continue
                    prev1 = df.iloc[loc-1]
                    if not prev1['green_candle'] or prev1['close'] < (prev1['high'] + prev1['low']) / 2: continue
                    if not (prev1['close'] > prev1[f"ema_{cfg['ema_period']}"]): continue
                    if not get_consecutive_red_candles(df, loc): continue
                    rs_ok = not cfg['rs_filter'] or (df.loc[date, 'return'] > index_df_daily.loc[date, 'return'] + cfg['rs_outperformance_pct'])
                    if not rs_ok: continue

                    today_candle = df.iloc[loc]
                    volume_ok = not cfg['volume_filter'] or (pd.notna(today_candle['volume_ma']) and today_candle['volume'] >= (today_candle['volume_ma'] * cfg['volume_multiplier']))
                    
                    entry_trigger_price = max([c['high'] for c in get_consecutive_red_candles(df, loc)] + [prev1['high']])
                    
                    # NOTE: For a live system, this check would be against the NEXT day's open/high.
                    # This example assumes we are generating signals for orders to be placed for the next day.
                    if volume_ok:
                        entry_price = entry_trigger_price
                        stop_loss = df.iloc[max(0, loc - cfg['stop_loss_lookback']):loc]['low'].min()
                        risk_per_share = entry_price - stop_loss
                        if risk_per_share <= 0: continue
                        
                        risk_capital = portfolio_equity * (cfg['risk_per_trade_percent'] / 100)
                        shares = math.floor(risk_capital / risk_per_share)
                        
                        if shares > 0 and (shares * entry_price) <= available_cash:
                            trade_plan['new_entries'].append({
                                'symbol': symbol,
                                'shares': shares,
                                'entry_price': round(entry_price, 2),
                                'stop_loss': round(stop_loss, 2),
                                'target': round(entry_price + risk_per_share, 2)
                            })
                except Exception: pass
        
        # Here you would add logic to check open positions for exits/modifications
        
        # --- Save Trade Plan to S3 ---
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=TRADE_PLAN_KEY,
            Body=json.dumps(trade_plan, indent=4)
        )
        print(f"Trade plan generated and saved to s3://{S3_BUCKET_NAME}/{TRADE_PLAN_KEY}")
        
        return {'statusCode': 200, 'body': json.dumps(trade_plan)}

    except Exception as e:
        print(f"An error occurred in signal generation: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps(f'Error: {str(e)}')}
