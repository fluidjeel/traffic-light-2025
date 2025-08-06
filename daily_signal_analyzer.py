# daily_signal_analyzer.py
#
# Description:
# An analysis script that ingests the output of the 'daily_entry_scanner.py'
# to determine the outcome of each entry signal. It calculates the initial risk
# and then traces the subsequent price action to see what Risk-to-Reward (R:R)
# multiple was achieved before the stop-loss would have been hit.
#
# MODIFICATION (v1.2 - Bug Fix):
# 1. FIXED: A ValueError in the final report generation that occurred when
#    trying to parse the string 'SL' as an integer. The logic now correctly
#    filters for R:R outcomes only.
#
# MODIFICATION (v1.1 - Pandas Compatibility Fix):
# 1. FIXED: A TypeError caused by using the 'method' argument in .get_loc(),
#    which is deprecated in newer pandas versions.

import pandas as pd
import os
import sys
import numpy as np
from collections import Counter

# --- CONFIGURATION ---
config = {
    # --- Input & Data Paths ---
    'log_folder': 'backtest_logs',
    'scanner_strategy_name': 'daily_entry_scanner',
    'scanner_output_filename': 'daily_entry_signals.csv',
    'processed_data_folder': os.path.join('data', 'universal_processed', 'daily'),
    'intraday_data_folder': os.path.join('data', 'universal_historical_data'),

    # --- Stop-Loss Definition (Must match the intended strategy) ---
    'stop_loss_mode': 'LOOKBACK',
    'stop_loss_lookback_days': 3,
    
    # --- Analysis Parameters ---
    'max_holding_days': 20, # How many days to track a signal before closing it
    'max_rr_target': 10,  # The maximum R:R multiple to check for
}

def analyze_signals(cfg):
    """
    Main function to run the signal analysis.
    """
    print("--- Starting Daily Signal R:R Pattern Analysis ---")

    # --- 1. Load Entry Signals ---
    signals_filepath = os.path.join(cfg['log_folder'], cfg['scanner_strategy_name'], cfg['scanner_output_filename'])
    if not os.path.exists(signals_filepath):
        print(f"ERROR: Signals file not found at: {signals_filepath}")
        print("Please run the 'daily_entry_scanner.py' script first.")
        return
        
    try:
        signals_df = pd.read_csv(signals_filepath, parse_dates=['setup_date', 'entry_timestamp'])
        print(f"Successfully loaded {len(signals_df)} entry signals.")
    except Exception as e:
        print(f"ERROR: Could not load signals file. Reason: {e}")
        return

    # --- 2. Load All Necessary Daily and Intraday Data ---
    print("Loading all required historical data...")
    daily_data, intraday_data = {}, {}
    unique_symbols = signals_df['symbol'].unique()
    for symbol in unique_symbols:
        try:
            daily_file = os.path.join(cfg['processed_data_folder'], f"{symbol}_daily_with_indicators.csv")
            if os.path.exists(daily_file):
                daily_data[symbol] = pd.read_csv(daily_file, index_col='datetime', parse_dates=True)
            
            intraday_file = os.path.join(cfg['intraday_data_folder'], f"{symbol}_15min.csv")
            if os.path.exists(intraday_file):
                intraday_data[symbol] = pd.read_csv(intraday_file, index_col='datetime', parse_dates=True)
        except Exception as e:
            print(f"Warning: Could not load all data for {symbol}. Error: {e}")
    print("Data loading complete.")

    # --- 3. Analyze Each Signal ---
    outcomes = []
    total_signals = len(signals_df)
    for idx, signal in signals_df.iterrows():
        progress_str = f"Analyzing Signal {idx + 1}/{total_signals}"
        sys.stdout.write(f"\r{progress_str.ljust(50)}"); sys.stdout.flush()

        symbol = signal['symbol']
        entry_price = signal['trigger_price'] # Use trigger as the theoretical entry
        entry_ts = signal['entry_timestamp']
        
        if symbol not in daily_data or symbol not in intraday_data:
            outcomes.append('Data Missing')
            continue

        # --- Calculate Initial Stop-Loss ---
        df_daily = daily_data[symbol]
        try:
            indexer = df_daily.index.get_indexer([entry_ts.date()], method='ffill')
            if indexer[0] == -1:
                outcomes.append('SL Calc Error')
                continue
            entry_day_loc = indexer[0]
            
            stop_loss_slice = df_daily.iloc[max(0, entry_day_loc - cfg['stop_loss_lookback_days']) : entry_day_loc]
            if stop_loss_slice.empty:
                outcomes.append('SL Calc Error')
                continue
            stop_loss_price = stop_loss_slice['low'].min()
        except (KeyError, IndexError):
            outcomes.append('SL Calc Error')
            continue

        initial_risk = entry_price - stop_loss_price
        if initial_risk <= 0:
            outcomes.append('Invalid Risk')
            continue

        # --- Define R:R Targets ---
        rr_targets = {f'Hit {i}R': entry_price + (initial_risk * i) for i in range(1, cfg['max_rr_target'] + 1)}
        
        # --- Trace Intraday Price Action ---
        df_intraday = intraday_data[symbol]
        trade_horizon = df_intraday.loc[entry_ts : entry_ts + pd.Timedelta(days=cfg['max_holding_days'])]
        
        final_outcome = 'Hit SL' # Default outcome is failure
        highest_rr_achieved = 0

        for _, candle in trade_horizon.iterrows():
            if candle['low'] <= stop_loss_price:
                break 

            for i in range(cfg['max_rr_target'], 0, -1):
                if i > highest_rr_achieved and candle['high'] >= rr_targets[f'Hit {i}R']:
                    highest_rr_achieved = i
                    break 
        
        if highest_rr_achieved > 0:
            final_outcome = f'Hit {highest_rr_achieved}R'
        
        outcomes.append(final_outcome)

    # --- 4. Generate Final Report ---
    print("\n\n--- R:R Pattern Analysis Complete ---")
    
    outcome_counts = Counter(outcomes)
    total_analyzed = len([o for o in outcomes if o.startswith('Hit')])

    print(f"\nTotal Valid Signals Analyzed: {total_analyzed}")
    print("-" * 40)
    print(f"{'Outcome':<15} | {'Count':>10} | {'Percentage':>10}")
    print("-" * 40)

    sl_count = outcome_counts.get('Hit SL', 0)
    sl_pct = (sl_count / total_analyzed) * 100 if total_analyzed > 0 else 0
    print(f"{'Hit SL':<15} | {sl_count:>10} | {sl_pct:>9.1f}%")

    # Print R:R results in order
    for i in range(1, cfg['max_rr_target'] + 1):
        # BUG FIX: Only parse keys that end in 'R' to avoid 'Hit SL'
        count_at_least_rr = sum(v for k, v in outcome_counts.items() if k.endswith('R') and int(k.split(' ')[1][:-1]) >= i)
        
        pct = (count_at_least_rr / total_analyzed) * 100 if total_analyzed > 0 else 0
        print(f"Achieved >{i-1}R {'':<5} | {count_at_least_rr:>10} | {pct:>9.1f}%")

    print("-" * 40)
    
    error_counts = {k: v for k, v in outcome_counts.items() if not k.startswith('Hit')}
    if error_counts:
        print("\nAnalysis Errors:")
        for error, count in error_counts.items():
            print(f"  - {error}: {count}")

if __name__ == "__main__":
    analyze_signals(config)
