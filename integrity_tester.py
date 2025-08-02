# integrity_tester.py
#
# Description:
# An independent script to perform a deep-dive integrity test on a single trade
# from the simulator's log files. It re-runs all the core logic for one trade
# to verify that the setup, entry, calculations, and exit were all handled
# correctly and without any form of bias or error.
#
# MODIFICATION (v2.1):
# - FIXED: A bug in the filename parsing logic that prevented the script from
#   finding the correct '_summary.txt' file. It now correctly reconstructs
#   the full timestamp (date and time).
#
# MODIFICATION (v2.0 - Truly Independent Auditor):
# - REMOVED: The script no longer imports the config from the simulator file.
# - ADDED: It now automatically finds the corresponding '_summary.txt' for a
#   trade log and dynamically parses it to reconstruct the exact config
#   used for that specific backtest run.

import pandas as pd
import os
import sys
import argparse
import glob
import re
import ast
from datetime import datetime

def parse_config_from_summary(summary_path):
    """
    Parses a _summary.txt file to dynamically reconstruct the config dictionary
    used for that specific backtest run.
    """
    config = {}
    with open(summary_path, 'r') as f:
        lines = f.readlines()

    capture = False
    for line in lines:
        line = line.strip()
        if line == "INPUT PARAMETERS:":
            capture = True
            continue
        if line.startswith("Final Equity:"):
            break
        if not capture or not line or line.startswith("---"):
            continue

        if ':' in line and not line.startswith('  -'):
            key, value = line.split(':', 1)
            key_formatted = key.replace(' ', '_').lower()
            
            # Handle nested dictionaries (like log_options)
            if value.strip() == "":
                config[key_formatted] = {}
            else:
                try:
                    # Try to evaluate the value as a Python literal
                    config[key_formatted] = ast.literal_eval(value.strip())
                except (ValueError, SyntaxError):
                    config[key_formatted] = value.strip()

        elif line.startswith('  -'):
            parent_key = list(config.keys())[-1]
            sub_key, sub_value = line.replace('  - ', '').split(':', 1)
            
            # Handle the list of dictionaries for vix_rr_scale
            if sub_key == 'vix_rr_scale':
                config[parent_key][sub_key] = ast.literal_eval(sub_value.strip())
            else:
                try:
                    config[parent_key][sub_key.strip()] = ast.literal_eval(sub_value.strip())
                except (ValueError, SyntaxError):
                    config[parent_key][sub_key.strip()] = sub_value.strip()
    
    # Manually correct the list of dicts for vix_rr_scale if parsing failed
    if 'vix_scaled_rr_config' in config and isinstance(config['vix_scaled_rr_config'].get('vix_rr_scale'), str):
        try:
            scale_str = config['vix_scaled_rr_config']['vix_rr_scale']
            config['vix_scaled_rr_config']['vix_rr_scale'] = ast.literal_eval(scale_str)
        except:
            print("Warning: Could not auto-parse vix_rr_scale.")

    return config


def load_all_data(config, symbol):
    """Loads all required data files for a given symbol."""
    data = {}
    try:
        # Load Monthly Data
        monthly_path = os.path.join(config['data_folder_base'], 'monthly', f"{symbol}_monthly_with_indicators.csv")
        data['monthly'] = pd.read_csv(monthly_path, index_col='datetime', parse_dates=True)
        
        # Load Daily Data
        daily_path = os.path.join(config['data_folder_base'], 'daily', f"{symbol}_daily_with_indicators.csv")
        data['daily'] = pd.read_csv(daily_path, index_col='datetime', parse_dates=True)

        # Load 15-Min Data
        intraday_path = os.path.join(config['intraday_data_folder'], f"{symbol}_15min.csv")
        data['intraday'] = pd.read_csv(intraday_path, index_col='datetime', parse_dates=True)

        # Load VIX Data
        vix_path = os.path.join(config['data_folder_base'], 'daily', f"{config['vix_symbol']}_daily_with_indicators.csv")
        data['vix'] = pd.read_csv(vix_path, index_col='datetime', parse_dates=True)
        
        print(f"✅ Successfully loaded all data files for {symbol}.")
        return data
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not load data file. {e}")
        return None

def run_integrity_test(setup_id):
    """
    Performs the step-by-step integrity audit for a single trade.
    """
    print(f"\n{'='*25} INTEGRITY TEST FOR SETUP: {setup_id} {'='*25}")
    
    # 1. Find the Trade and its Corresponding Config
    # --------------------------------------------------
    print("\n--- [STEP 1: FINDING TRADE AND CONFIG] ---")
    
    # Find the trade log that contains the setup_id
    strategy_log_folder = 'backtest_logs' 
    all_trade_files = glob.glob(os.path.join(strategy_log_folder, '*', '*_trade_details.csv'))
    
    source_log_file = None
    trade_to_audit = None

    for f in all_trade_files:
        try:
            df = pd.read_csv(f)
            if setup_id in df['setup_id'].values:
                trade_to_audit = df[df['setup_id'] == setup_id].iloc[0]
                source_log_file = f
                break
        except (pd.errors.EmptyDataError, KeyError):
            continue

    if trade_to_audit is None:
        print(f"❌ ERROR: Could not find a trade with setup_id '{setup_id}' in any log file.")
        return
        
    print(f"  - Found trade in log file: {os.path.basename(source_log_file)}")
    
    # --- BUG FIX: Correctly parse the full timestamp from the filename ---
    filename_parts = os.path.basename(source_log_file).split('_')
    if len(filename_parts) >= 2:
        timestamp = f"{filename_parts[0]}_{filename_parts[1]}"
    else:
        print("❌ ERROR: Could not parse timestamp from log file name.")
        return
    # --- END BUG FIX ---
    
    summary_file_path = os.path.join(os.path.dirname(source_log_file), f"{timestamp}_summary.txt")
    
    if not os.path.exists(summary_file_path):
        print(f"❌ ERROR: Corresponding summary file not found at '{summary_file_path}'.")
        return
        
    config = parse_config_from_summary(summary_file_path)
    print(f"  - Successfully parsed historical config from: {os.path.basename(summary_file_path)}")

    # 2. Load Data
    # --------------------------------------------------
    print("\n--- [STEP 2: LOADING DATA] ---")
    symbol, date_str = setup_id.rsplit('_', 1)
    all_data = load_all_data(config, symbol)
    if not all_data:
        return

    # 3. Verify Scout Logic (The Setup)
    # --------------------------------------------------
    print("\n--- [STEP 3: VERIFYING SCOUT SETUP LOGIC] ---")
    setup_date = pd.to_datetime(date_str)
    monthly_df = all_data['monthly']
    setup_candle = monthly_df.loc[setup_date]
    loc = monthly_df.index.get_loc(setup_date)
    prev_candle = monthly_df.iloc[loc - 1]
    
    is_green = setup_candle['close'] > setup_candle['open']
    prev_is_red = prev_candle['close'] < prev_candle['open']
    print(f"  - Setup Candle ({setup_date.date()}): Green? -> {is_green}")
    print(f"  - Previous Candle ({prev_candle.name.date()}): Red? -> {prev_is_red}")
    if is_green and prev_is_red: print("✅ VERDICT: Setup pattern is confirmed.")
    else: print("❌ FLAW DETECTED: Invalid monthly pattern."); return
    
    trigger_price = max(setup_candle['high'], prev_candle['high'])
    print(f"  - Calculated Trigger Price: {trigger_price:.2f}")

    # 4. Verify Sniper Logic & Calculations
    # --------------------------------------------------
    print("\n--- [STEP 4: VERIFYING SNIPER LOGIC & CALCULATIONS] ---")
    trade = trade_to_audit
    entry_datetime = pd.to_datetime(trade['entry_date'])
    print(f"  - Entry Price (from log): {trade['entry_price']:.2f}")
    print(f"  - Stop-Loss (from log): {trade['stop_loss']:.2f}")
    print(f"  - Target (from log): {trade['target']:.2f}")

    # Re-calculate Stop-Loss
    recalculated_stop = trade['entry_price'] * (1 - config['fixed_stop_loss_percent'])
    print(f"  - Recalculated Stop-Loss: {recalculated_stop:.2f}")
    if abs(recalculated_stop - trade['stop_loss']) < 0.01:
        print("✅ VERDICT: Stop-loss calculation is confirmed.")
    else:
        print("❌ FLAW DETECTED: Stop-loss calculation does NOT match.")

    # Re-calculate Profit Target
    vix_df = all_data['vix']
    day_before_entry = entry_datetime.date() - pd.Timedelta(days=1)
    vix_close_indexer = vix_df.index.get_indexer([day_before_entry], method='ffill')
    vix_value = vix_df.iloc[vix_close_indexer[0]]['close']
    
    risk_per_share = trade['entry_price'] - trade['stop_loss']
    
    profit_target_rr = -1
    rr_config = config.get('vix_scaled_rr_config', {})
    if rr_config.get('use_vix_scaled_rr_target', False):
        scale = rr_config.get('vix_rr_scale', [])
        profit_target_rr = scale[-1]['rr']
        for level in scale:
            if vix_value <= level['vix_max']:
                profit_target_rr = level['rr']
                break
    else:
        profit_target_rr = config['profit_target_rr_calm'] if vix_value <= config['vix_high_threshold'] else config['profit_target_rr_high']
            
    recalculated_target = trade['entry_price'] + (risk_per_share * profit_target_rr)
    print(f"  - VIX on day before entry: {vix_value:.2f}, resulting in R:R multiple of {profit_target_rr}")
    print(f"  - Recalculated Target: {recalculated_target:.2f}")
    if abs(recalculated_target - trade['target']) < 0.01:
        print("✅ VERDICT: Profit target calculation is confirmed.")
    else:
        print("❌ FLAW DETECTED: Profit target calculation does NOT match.")

    # 5. Verify Exit Logic
    # --------------------------------------------------
    print("\n--- [STEP 5: VERIFYING EXIT LOGIC] ---")
    intraday_df = all_data['intraday']
    exit_datetime = pd.to_datetime(trade['exit_date'])
    trade_period_data = intraday_df.loc[(intraday_df.index >= entry_datetime) & (intraday_df.index <= exit_datetime)]
    
    lowest_low_during_trade = trade_period_data['low'].min()
    highest_high_during_trade = trade_period_data['high'].max()

    print(f"  - Lowest low during trade: {lowest_low_during_trade:.2f}")
    print(f"  - Highest high during trade: {highest_high_during_trade:.2f}")

    exit_type = trade['exit_type']
    print(f"  - Exit Type (from log): {exit_type}")

    if exit_type == 'Profit Target':
        if highest_high_during_trade >= trade['target']:
             print("✅ VERDICT: Profit Target exit is confirmed.")
        else:
             print("❌ FLAW DETECTED: Price never hit the profit target.")
    elif exit_type == 'Stop-Loss':
        if lowest_low_during_trade <= trade['stop_loss']:
             print("✅ VERDICT: Stop-Loss exit is confirmed.")
        else:
             print("❌ FLAW DETECTED: Price never hit the stop-loss.")
             
    print(f"\n{'='*25} INTEGRITY TEST COMPLETE {'='*25}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent integrity tester for a single trade.")
    parser.add_argument('--setup_id', type=str, required=True, help="The 'setup_id' of the trade to audit from the trade log (e.g., 'RELIANCE_2023-03-31').")
    args = parser.parse_args()
    
    run_integrity_test(args.setup_id)
