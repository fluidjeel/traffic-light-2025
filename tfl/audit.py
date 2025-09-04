import pandas as pd
import os
import re
import numpy as np
import warnings
import datetime

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# This script should be run from the root of your project (e.g., D:\algo-2025)
ROOT_DIR = os.getcwd()
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")

# The script now reads from the two separate data files.
LONGS_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_longs_data_with_signals_and_atr.parquet")
SHORTS_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_shorts_data_with_signals_and_atr.parquet")

# ==============================================================================
# --- CORE AUDIT LOGIC ---
# ==============================================================================

def find_latest_run_dir(logs_base_dir):
    """Finds the most recent backtest run directory."""
    latest_run_dir = None
    latest_time = None

    if not os.path.exists(logs_base_dir):
        return None

    for strategy_folder in os.listdir(logs_base_dir):
        strategy_path = os.path.join(logs_base_dir, strategy_folder)
        if os.path.isdir(strategy_path):
            for run_folder in os.listdir(strategy_path):
                run_path = os.path.join(strategy_path, run_folder)
                if os.path.isdir(run_path):
                    try:
                        run_time = datetime.datetime.strptime(run_folder, '%Y%m%d_%H%M%S')
                        if latest_time is None or run_time > latest_time:
                            latest_time = run_time
                            latest_run_dir = run_path
                    except ValueError:
                        continue
    return latest_run_dir


def parse_config_from_summary(summary_path):
    """Parses key parameters from the summary.txt file."""
    config = {"longs": {}, "shorts": {}}
    if not os.path.exists(summary_path):
        return config
        
    with open(summary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_profile = None
    for line in lines:
        if "--- LONG STRATEGY PROFILE ---" in line:
            current_profile = "longs"
        elif "--- SHORT STRATEGY PROFILE ---" in line:
            current_profile = "shorts"
        
        if current_profile:
            match = re.search(r'-\s*([\w_]+):\s*([\d\.]+)', line)
            if match:
                key, value = match.groups()
                config[current_profile][key] = float(value)
    
    return config


def run_audit(latest_run_dir, data_df):
    """Performs the full audit on the latest backtest run."""
    print(f"\n--- Starting Audit for Run: {os.path.basename(os.path.dirname(latest_run_dir))}/{os.path.basename(latest_run_dir)} ---")
    
    trade_log_path = os.path.join(latest_run_dir, "trade_log.csv")
    summary_path = os.path.join(latest_run_dir, "summary.txt")

    if not os.path.exists(trade_log_path):
        print("Audit SKIPPED: trade_log.csv not found.")
        return

    trade_log_df = pd.read_csv(trade_log_path, parse_dates=['entry_time', 'exit_time'])
    config = parse_config_from_summary(summary_path)
    
    # --- CRITICAL TIMEZONE FIX ---
    # The trade log timestamps are read with a fixed offset (+05:30). We must
    # convert them to the named 'Asia/Kolkata' timezone to ensure a perfect
    # match with the source data's index for the merge operation.
    try:
        trade_log_df['entry_time'] = trade_log_df['entry_time'].dt.tz_convert('Asia/Kolkata')
        trade_log_df['exit_time'] = trade_log_df['exit_time'].dt.tz_convert('Asia/Kolkata')
    except TypeError:
        # Handle cases where timestamps might be naive
        trade_log_df['entry_time'] = pd.to_datetime(trade_log_df['entry_time']).dt.tz_localize('Asia/Kolkata', ambiguous='infer')
        trade_log_df['exit_time'] = pd.to_datetime(trade_log_df['exit_time']).dt.tz_localize('Asia/Kolkata', ambiguous='infer')

    
    merged_trades = pd.merge(trade_log_df, data_df.reset_index(), 
                             left_on=['entry_time', 'symbol', 'direction'], 
                             right_on=['datetime', 'symbol', 'direction'],
                             how='left', suffixes=('', '_entry_candle'))

    audit_failures = []
    total_trades = len(trade_log_df)
    print(f"Loaded {total_trades} trades. Auditing now...")

    for i, trade in merged_trades.iterrows():
        # --- 1. Signal Presence Check ---
        if pd.isna(trade['is_fast_entry']) or not trade['is_fast_entry']:
            audit_failures.append({**trade.to_dict(), "failure_reason": "Signal not present in data file"})
            continue

        # --- 2. Entry Price Feasibility ---
        if not (trade['low'] <= trade['entry_price'] <= trade['high']):
            audit_failures.append({**trade.to_dict(), "failure_reason": "Entry price outside of candle range"})
            
        # --- 3. Exit Price Feasibility ---
        exit_candle_slice = data_df[(data_df['symbol'] == trade['symbol']) & (data_df.index == trade['exit_time'])]
        if not exit_candle_slice.empty:
            exit_candle = exit_candle_slice.iloc[0]
            if not (exit_candle['low'] <= trade['exit_price'] <= exit_candle['high']):
                if trade['exit_reason'] not in ['GAP_SL_HIT', 'GAP_TP_HIT']:
                     audit_failures.append({**trade.to_dict(), "failure_reason": "Exit price outside of candle range"})
        
        # --- 4. SL Placement Check ---
        if trade['direction'] == 'LONG' and not np.isclose(trade['initial_sl'], trade['pattern_low_for_sl']):
             audit_failures.append({**trade.to_dict(), "failure_reason": "Incorrect SL for LONG (not pattern_low)"})
        elif trade['direction'] == 'SHORT' and not np.isclose(trade['initial_sl'], trade['pattern_high_for_sl']):
             audit_failures.append({**trade.to_dict(), "failure_reason": "Incorrect SL for SHORT (not pattern_high)"})

        # --- 5. Trading Cost & P&L Verification ---
        expected_pnl = 0
        if trade['direction'] == 'LONG':
            txn_cost_pct = config.get('longs', {}).get('transaction_cost_pct', 0.0)
            net_proceeds = (trade['quantity'] * trade['exit_price']) * (1 - txn_cost_pct)
            expected_pnl = net_proceeds - trade['initial_cost_with_fees']
        elif trade['direction'] == 'SHORT':
            txn_cost_pct = config.get('shorts', {}).get('transaction_cost_pct', 0.0)
            cost_to_cover = (trade['quantity'] * trade['exit_price']) * (1 + txn_cost_pct)
            expected_pnl = trade['initial_proceeds'] - cost_to_cover
            
        if not np.isclose(expected_pnl, trade['pnl']):
            audit_failures.append({**trade.to_dict(), "failure_reason": f"P&L Mismatch (Expected: {expected_pnl:.2f}, Actual: {trade['pnl']:.2f})"})

    # --- Final Report ---
    if not audit_failures:
        print("\n--- AUDIT PASSED ---")
        print("All trades were validated successfully. No logical or calculation flaws found.")
    else:
        print(f"\n--- AUDIT FAILED: Found {len(audit_failures)} discrepancies. ---")
        failures_df = pd.DataFrame(audit_failures)
        failure_log_path = os.path.join(latest_run_dir, "audit_failures.csv")
        failures_df.to_csv(failure_log_path, index=False)
        print(f"A detailed log of all failures has been saved to: {failure_log_path}")

def main():
    """Main function to orchestrate the audit process."""
    latest_run = find_latest_run_dir(LOGS_BASE_DIR)
    
    if not latest_run:
        print(f"ERROR: No backtest log directories found in '{LOGS_BASE_DIR}'.")
        return

    print("Loading and unifying long and short data files for cross-verification...")
    
    try:
        if not os.path.exists(LONGS_DATA_PATH):
            print(f"ERROR: Longs data file not found at '{LONGS_DATA_PATH}'.")
            return
        longs_df = pd.read_parquet(LONGS_DATA_PATH)
        longs_df['direction'] = 'LONG'
        longs_df.reset_index(inplace=True)
        
        if not os.path.exists(SHORTS_DATA_PATH):
            print(f"ERROR: Shorts data file not found at '{SHORTS_DATA_PATH}'.")
            return
        shorts_df = pd.read_parquet(SHORTS_DATA_PATH)
        shorts_df['direction'] = 'SHORT'
        shorts_df.reset_index(inplace=True)
        
        unified_df = pd.concat([longs_df, shorts_df], ignore_index=True)
        
        unified_df.sort_values(by=['symbol', 'datetime'], inplace=True)
        unified_df['pattern_low_for_sl'] = unified_df.groupby('symbol')['pattern_low'].shift(1)
        unified_df['pattern_high_for_sl'] = unified_df.groupby('symbol')['pattern_high'].shift(1)
        
        unified_df['datetime'] = pd.to_datetime(unified_df['datetime'], utc=True).dt.tz_convert('Asia/Kolkata')
        unified_df.set_index('datetime', inplace=True)
        unified_df.sort_index(inplace=True)
        
    except Exception as e:
        print(f"An error occurred while loading data files: {e}")
        return

    run_audit(latest_run, unified_df)

if __name__ == "__main__":
    main()

