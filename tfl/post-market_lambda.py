import boto3
import os
import json
from datetime import datetime, timedelta
from decimal import Decimal

# ==============================================================================
# --- LAMBDA CONFIGURATION & ENVIRONMENT VARIABLES ---
# ==============================================================================
# This Lambda function requires the following environment variables to be set:
#
# DYNAMODB_TABLE_NAME: The name of the DynamoDB table storing trade data.
# SNS_TOPIC_ARN_REPORTS: The ARN of the SNS topic for sending daily reports.

AWS_REGION = "ap-south-1"

# ==============================================================================
# --- AWS CLIENTS (initialized globally for reuse) ---
# ==============================================================================
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
sns_client = boto3.client('sns', region_name=AWS_REGION)

def lambda_handler(event, context):
    """
    This is the main entry point. It queries DynamoDB for today's closed trades,
    calculates performance metrics, and sends a report via SNS.
    """
    print("--- Starting Post-Market Reporting ---")

    # --- Load Configuration from Environment Variables ---
    table_name = os.environ['DYNAMODB_TABLE_NAME']
    sns_topic = os.environ['SNS_TOPIC_ARN_REPORTS']

    table = dynamodb.Table(table_name)
    
    # --- 1. Define Date Range for Today's Trades ---
    # We need to query for trades closed since the last market open.
    # This is a simplified approach. A production system would use a GSI.
    utc_now = datetime.utcnow()
    utc_yesterday = utc_now - timedelta(days=1)
    
    # In a production system with high trade volume, you would query a GSI
    # on the 'status' and 'exit_timestamp_utc' attributes.
    # A scan is acceptable for low-to-moderate trade volumes.
    try:
        response = table.scan(
            FilterExpression="attribute_exists(exit_timestamp_utc) AND #s = :closed",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":closed": "CLOSED"}
        )
        all_closed_trades = response.get('Items', [])
        
        # Filter for trades closed within the last 24 hours
        todays_trades = [
            trade for trade in all_closed_trades 
            if datetime.fromisoformat(trade['exit_timestamp_utc']) > utc_yesterday
        ]

    except Exception as e:
        message = f"FATAL: Could not scan DynamoDB table '{table_name}'. Error: {e}"
        print(message)
        sns_client.publish(TopicArn=sns_topic, Message=message, Subject="TFL Reporting Lambda ERROR")
        return {'statusCode': 500, 'body': json.dumps(message)}

    # --- 2. Calculate Performance Metrics ---
    if not todays_trades:
        report_body = "No trades were closed today."
    else:
        total_trades = len(todays_trades)
        pnl_values = [trade.get('pnl', 0) for trade in todays_trades]
        
        # Convert Decimal types from DynamoDB to float for calculation
        pnl_values = [float(p) for p in pnl_values]
        
        wins = [p for p in pnl_values if p > 0]
        losses = [p for p in pnl_values if p <= 0]
        
        total_pnl = sum(pnl_values)
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        
        # --- 3. Format the Report ---
        report_body = (
            f"--- TFL Daily Performance Report ---\n\n"
            f"Total Trades Closed Today: {total_trades}\n"
            f"Winning Trades: {len(wins)}\n"
            f"Losing Trades: {len(losses)}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"------------------------------------\n"
            f"Net PnL for Today: Rs. {total_pnl:,.2f}\n"
        )

    # --- 4. Send Report via SNS ---
    try:
        print("Sending daily report via SNS...")
        sns_client.publish(
            TopicArn=sns_topic,
            Message=report_body,
            Subject=f"TFL Trading Report - {datetime.now(INDIA_TZ).strftime('%Y-%m-%d')}"
        )
        print("Report sent successfully.")
    except Exception as e:
        print(f"FATAL: Failed to send SNS notification. Error: {e}")
        # Still return 200 so the Lambda doesn't retry on SNS failure
        return {'statusCode': 500, 'body': json.dumps(str(e))}

    return {
        'statusCode': 200,
        'body': json.dumps("Report generated and sent successfully.")
    }
