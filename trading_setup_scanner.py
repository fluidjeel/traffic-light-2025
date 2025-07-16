# trading_setup_scanner.py
#
# Description:
# This script scans through downloaded historical data (daily, weekly, monthly)
# to identify specific long-only trading entry setups and saves the results
# to a timestamped CSV file. The EMA period used as a filter is configurable
# for each timeframe.
#
# Prerequisites:
# 1. You must have folders containing the historical data with all necessary
#    indicator columns (e.g., 'EMA_20', 'EMA_30').
#    - 'daily_with_indicators'
#    - 'weekly_with_indicators'
#    - 'monthly_with_indicators'
# 2. Required libraries installed: pip install pandas
#
# How to use:
# 1. Ensure the data folders are populated and contain the required EMA columns.
# 2. Configure the `scan_plan` dictionary below to set your desired EMA for each timeframe.
# 3. Run this script. It will save any found setups to a file.

import os
import pandas as pd
from datetime import datetime

# --- Configuration ---
# A small buffer to be subtracted from the low for setting the stop-loss.
STOP_LOSS_BUFFER_PERCENT = 0.010

# --- Strategy Functions ---

def check_daily_entry(df, ema_period):
    """
    Checks for the Daily Entry setup.
    Condition: A red candle is immediately followed by a green candle,
               and the price is above the specified EMA.
    """
    ema_column = f'EMA_{ema_period}'
    if len(df) < 5 or ema_column not in df.columns:
        return None

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    # Conditions
    is_red_candle = prev_candle['close'] < prev_candle['open']
    is_green_candle = last_candle['close'] > last_candle['open']
    price_above_ema = last_candle['close'] > last_candle[ema_column]

    if is_red_candle and is_green_candle and price_above_ema:
        entry_price = max(last_candle['high'], prev_candle['high'])
        lowest_low_5 = df['low'].tail(5).min()
        stop_loss = lowest_low_5 * (1 - STOP_LOSS_BUFFER_PERCENT)
        
        return {
            "setup_type": f"Daily Entry (EMA {ema_period})",
            "entry_price": round(entry_price, 2),
            "stop_loss_price": round(stop_loss, 2)
        }
    return None

def check_weekly_entry(df, ema_period):
    """
    Checks for the Weekly Entry setup.
    Condition: A red candle is immediately followed by a green candle,
               and the price is above the specified EMA.
    """
    ema_column = f'EMA_{ema_period}'
    if len(df) < 2 or ema_column not in df.columns:
        return None

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    # Conditions
    is_red_candle = prev_candle['close'] < prev_candle['open']
    is_green_candle = last_candle['close'] > last_candle['open']
    price_above_ema = last_candle['close'] > last_candle[ema_column]

    if is_red_candle and is_green_candle and price_above_ema:
        entry_price = max(last_candle['high'], prev_candle['high'])
        stop_loss = prev_candle['low'] * (1 - STOP_LOSS_BUFFER_PERCENT)
        
        return {
            "setup_type": f"Weekly Entry (EMA {ema_period})",
            "entry_price": round(entry_price, 2),
            "stop_loss_price": round(stop_loss, 2)
        }
    return None

def check_weekly_immediate_entry(df, ema_period):
    """
    Checks for the Weekly Immediate Entry setup.
    Condition: Red candle followed by green, price is above specified EMA, and
               green candle's high has not yet crossed the red candle's high.
    """
    ema_column = f'EMA_{ema_period}'
    if len(df) < 2 or ema_column not in df.columns:
        return None

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    # Conditions
    is_red_candle = prev_candle['close'] < prev_candle['open']
    is_green_candle = last_candle['close'] > last_candle['open']
    price_above_ema = last_candle['close'] > last_candle[ema_column]
    high_not_crossed = last_candle['high'] < prev_candle['high']

    if is_red_candle and is_green_candle and high_not_crossed and price_above_ema:
        entry_price = prev_candle['high']
        stop_loss = prev_candle['low'] * (1 - STOP_LOSS_BUFFER_PERCENT)
        
        return {
            "setup_type": f"Weekly Immediate Entry (EMA {ema_period})",
            "entry_price": round(entry_price, 2),
            "stop_loss_price": round(stop_loss, 2)
        }
    return None

def check_monthly_entry(df, ema_period):
    """
    Checks for the Monthly Entry setup. Logic is the same as Weekly Entry.
    """
    setup = check_weekly_entry(df, ema_period)
    if setup:
        setup["setup_type"] = f"Monthly Entry (EMA {ema_period})"
        return setup
    return None

def check_monthly_immediate_entry(df, ema_period):
    """
    Checks for the Monthly Immediate Entry setup. Logic is the same as Weekly Immediate Entry.
    """
    setup = check_weekly_immediate_entry(df, ema_period)
    if setup:
        setup["setup_type"] = f"Monthly Immediate Entry (EMA {ema_period})"
        return setup
    return None

def scan_data_for_setups():
    """
    Main function to loop through all data files, check for setups,
    and save the results to a CSV file.
    """
    # --- SCAN PLAN CONFIGURATION ---
    # Define which checks to run on which directories and with which EMA period.
    # Make sure the corresponding EMA_{period} column exists in your data files.
    scan_plan = {
        "daily_with_indicators": {
            "ema_period": 30,
            "check_functions": [check_daily_entry]
        },
        "weekly_with_indicators": {
            "ema_period": 30,
            "check_functions": [check_weekly_entry, check_weekly_immediate_entry]
        },
        "monthly_with_indicators": {
            "ema_period": 30,
            "check_functions": [check_monthly_entry, check_monthly_immediate_entry]
        }
    }

    print("--- Starting Trading Setup Scan ---")
    
    all_alerts = []

    for directory, config in scan_plan.items():
        if not os.path.isdir(directory):
            print(f"\nWarning: Directory '{directory}' not found. Skipping.")
            continue

        print(f"\nScanning directory: '{directory}' using EMA_{config['ema_period']}...")
        
        data_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if not data_files:
            print("  No data files found.")
            continue

        for filename in data_files:
            symbol_name = filename.split('_')[0]
            file_path = os.path.join(directory, filename)
            
            try:
                df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
                
                # Run all check functions for this timeframe
                for check_func in config['check_functions']:
                    result = check_func(df, config['ema_period'])
                    if result:
                        # Add the stock symbol to the results dictionary
                        result['symbol'] = symbol_name
                        all_alerts.append(result)
                        print(f"  > Setup found for {symbol_name}: {result['setup_type']}")

            except Exception as e:
                print(f"  > Error processing {filename}: {e}")

    # --- Save results to a file ---
    if all_alerts:
        # Create a DataFrame from the list of alerts
        alerts_df = pd.DataFrame(all_alerts)
        
        # Reorder columns for better readability
        alerts_df = alerts_df[['symbol', 'setup_type', 'entry_price', 'stop_loss_price']]

        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_filename = f"trading_setups_{timestamp}.csv"
        
        # Save to CSV
        alerts_df.to_csv(output_filename, index=False)
        print(f"\n*** Scan Complete: {len(all_alerts)} setups found and saved to '{output_filename}' ***")
    else:
        print("\n--- Scan Complete: No trading setups found. ---")


if __name__ == "__main__":
    scan_data_for_setups()
