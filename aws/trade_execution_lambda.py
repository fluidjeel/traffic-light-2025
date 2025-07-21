import json
import os
import boto3
from fyers_apiv3 import fyersModel
from datetime import datetime

# --- Environment Variables ---
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
TOKEN_SECRET_NAME = os.environ.get('TOKEN_SECRET_NAME')
CLIENT_ID = os.environ.get('FYERS_CLIENT_ID')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')
EXECUTION_MODE = os.environ.get('EXECUTION_MODE', 'PAPER') # Defaults to PAPER for safety
PAPER_TRADE_OPEN_KEY = 'paper_trades_open.json' # S3 key for open paper trades

# --- AWS Clients ---
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')

def get_access_token_from_secrets():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=TOKEN_SECRET_NAME)
    secret = json.loads(response['SecretString'])
    return secret.get('access_token')

def lambda_handler(event, context):
    print(f"Trade Execution Lambda started in {EXECUTION_MODE} mode.")
    
    try:
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        trade_plan = json.loads(obj['Body'].read().decode('utf-8'))
        print(f"Loaded trade plan from s3://{bucket}/{key}")

        fyers = None
        if EXECUTION_MODE == 'LIVE':
            access_token = get_access_token_from_secrets()
            if not access_token: raise Exception("Could not retrieve access token.")
            fyers = fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=access_token)
            print("Fyers API client initialized for LIVE trading.")

        execution_summary = []

        if EXECUTION_MODE == 'LIVE':
            # --- LIVE TRADING LOGIC ---
            for entry in trade_plan.get('new_entries', []):
                # ... (live order placement logic remains the same) ...
                pass # Placeholder for brevity
        else:
            # --- PAPER TRADING LOGIC ---
            if trade_plan.get('new_entries'):
                try:
                    # Get existing open paper trades
                    obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=PAPER_TRADE_OPEN_KEY)
                    open_trades = json.loads(obj['Body'].read().decode('utf-8'))
                except s3_client.exceptions.NoSuchKey:
                    open_trades = [] # Create a new list if the file doesn't exist

                for entry in trade_plan['new_entries']:
                    # Add a unique ID and the entry date to the trade
                    entry['trade_id'] = f"{entry['symbol']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    entry['entry_date'] = datetime.now().strftime('%Y-%m-%d')
                    open_trades.append(entry)
                    msg = f"PAPER TRADE LOGGED: {entry['shares']} shares of {entry['symbol']} at ~{entry['entry_price']}."
                    print(msg)
                    execution_summary.append(msg)
                
                # Save the updated list of open paper trades back to S3
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=PAPER_TRADE_OPEN_KEY,
                    Body=json.dumps(open_trades, indent=4)
                )
                print(f"Updated open paper trades file in S3.")

        final_message = f"Trading Execution Report ({EXECUTION_MODE} Mode):\n\n" + "\n".join(execution_summary)
        if not execution_summary:
            final_message = f"Trading Execution Report ({EXECUTION_MODE} Mode): No new actions were required today."
            
        sns_client.publish(TopicArn=SNS_TOPIC_ARN, Message=final_message, Subject=f"Daily Trading Strategy Report ({EXECUTION_MODE} Mode)")
        print("Final notification sent via SNS.")

        return {'statusCode': 200, 'body': json.dumps('Execution complete.')}

    except Exception as e:
        print(f"An error occurred in trade execution: {str(e)}")
        sns_client.publish(TopicArn=SNS_TOPIC_ARN, Message=f"CRITICAL ERROR in Trade Execution Lambda: {str(e)}", Subject=f"Trading System FAILURE ({EXECUTION_MODE} Mode)")
        return {'statusCode': 500, 'body': json.dumps(f'Error: {str(e)}')}
