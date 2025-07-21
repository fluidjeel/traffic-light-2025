import json
import os
import boto3
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime

# --- Environment Variables ---
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')
PAPER_TRADE_OPEN_KEY = 'paper_trades_open.json'
PAPER_TRADE_CLOSED_KEY = 'paper_trades_closed.csv'
DATA_FOLDER_DAILY = 'data/processed/daily'

# --- AWS Clients ---
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')

def lambda_handler(event, context):
    """
    Manages and journals open paper trades on a daily basis.
    """
    print("Paper Trade Journaling Lambda started.")
    
    try:
        # 1. Load open paper trades from S3
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=PAPER_TRADE_OPEN_KEY)
            open_trades = json.loads(obj['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            print("No open paper trades file found. Nothing to process.")
            return {'statusCode': 200, 'body': json.dumps('No open paper trades.')}

        if not open_trades:
            print("Open paper trades file is empty. Nothing to process.")
            return {'statusCode': 200, 'body': json.dumps('No open paper trades.')}

        # 2. Load closed trades log to append to it
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=PAPER_TRADE_CLOSED_KEY)
            closed_trades_log = obj['Body'].read().decode('utf-8')
        except s3_client.exceptions.NoSuchKey:
            closed_trades_log = "trade_id,symbol,entry_date,exit_date,entry_price,exit_price,pnl,exit_type,shares\n"

        # 3. Simulate today's price action against open trades
        remaining_open_trades = []
        closed_today_summary = []
        today_date = datetime.now().date()

        for trade in open_trades:
            symbol = trade['symbol']
            is_trade_closed = False
            
            try:
                # Load the latest daily data for the symbol
                data_key = f"{DATA_FOLDER_DAILY}/{symbol}_daily_with_indicators.csv"
                obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=data_key)
                df = pd.read_csv(BytesIO(obj['Body'].read()), index_col=0, parse_dates=True)
                df.columns = [col.lower() for col in df.columns]
                
                # Get today's candle (or the last available one)
                today_candle = df.iloc[-1]

                # Check for partial profit
                if not trade.get('partial_exit', False) and today_candle['high'] >= trade['target']:
                    exit_price = trade['target']
                    pnl = (exit_price - trade['entry_price']) * (trade['shares'] / 2)
                    closed_trades_log += f"{trade['trade_id']}_p1,{symbol},{trade['entry_date']},{today_date},{trade['entry_price']},{exit_price},{pnl},Partial Profit,{trade['shares']/2}\n"
                    closed_today_summary.append(f"Partial Profit on {symbol}: P&L {pnl:.2f}")
                    
                    trade['shares'] /= 2
                    trade['stop_loss'] = trade['entry_price']
                    trade['partial_exit'] = True

                # Check for stop loss
                if trade['shares'] > 0 and today_candle['low'] <= trade['stop_loss']:
                    exit_price = trade['stop_loss']
                    pnl = (exit_price - trade['entry_price']) * trade['shares']
                    closed_trades_log += f"{trade['trade_id']}_p2,{symbol},{trade['entry_date']},{today_date},{trade['entry_price']},{exit_price},{pnl},Stop-Loss,{trade['shares']}\n"
                    closed_today_summary.append(f"Stop-Loss on {symbol}: P&L {pnl:.2f}")
                    is_trade_closed = True

                # Trail the stop if still open
                if not is_trade_closed and today_candle['close'] > trade['entry_price']:
                    trade['stop_loss'] = max(trade['stop_loss'], trade['entry_price'])
                    if today_candle['close'] > today_candle['open']: # Green candle
                        trade['stop_loss'] = max(trade['stop_loss'], today_candle['low'])
                
                if not is_trade_closed:
                    remaining_open_trades.append(trade)

            except Exception as e:
                print(f"Could not process trade {trade['trade_id']} for {symbol}. Error: {e}. Keeping it open.")
                remaining_open_trades.append(trade)

        # 4. Save updated files back to S3
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=PAPER_TRADE_OPEN_KEY, Body=json.dumps(remaining_open_trades, indent=4))
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=PAPER_TRADE_CLOSED_KEY, Body=closed_trades_log)
        
        print("Paper trade journaling complete.")
        
        # 5. Send notification
        final_message = "Paper Trading Daily Journaling Report:\n\n" + "\n".join(closed_today_summary)
        if not closed_today_summary:
            final_message = "Paper Trading Daily Journaling Report: No paper trades were closed today."
        
        sns_client.publish(TopicArn=SNS_TOPIC_ARN, Message=final_message, Subject="Daily Paper Trading Journal Report")

        return {'statusCode': 200, 'body': json.dumps('Journaling complete.')}

    except Exception as e:
        print(f"An error occurred in paper trade journaling: {str(e)}")
        sns_client.publish(TopicArn=SNS_TOPIC_ARN, Message=f"CRITICAL ERROR in Paper Trade Journaling Lambda: {str(e)}", Subject="Trading System FAILURE")
        return {'statusCode': 500, 'body': json.dumps(f'Error: {str(e)}')}
