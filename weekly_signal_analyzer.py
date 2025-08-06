# weekly_signal_analyzer.py
#
# Description:
# An analysis script that ingests the output of the 'weekly_entry_scanner.py'
# to determine the outcome of each weekly entry signal. It calculates the
# initial risk and then traces the subsequent weekly price action to see
# what Risk-to-Reward (R:R) multiple was achieved before the stop-loss would
# have been hit.

import pandas as pd
import os
import sys
import numpy as np
from collections import Counter

# --- CONFIGURATION ---
config = {
    # --- Input & Data Paths ---
    'log_folder': 'backtest_logs',
    'scanner_strategy_name': 'weekly_entry_scanner',
    'scanner_output_filename': 'weekly_entry_signals.csv',
    'processed_data_folder': os.path.join('data', 'universal_processed', 'weekly'), # Use weekly processed data
    # No intraday data needed for this analyzer, as it operates on weekly candles.

    # --- Stop-Loss Definition (Must match the intended strategy) ---
    'stop_loss_mode': 'LOOKBACK',
    'stop_loss_lookback_weeks': 2, # Stop-loss lookback based on weekly candles
    
    # --- Analysis Parameters ---
    'max_holding_weeks': 10, # How many weeks to track a signal before closing it
    'max_rr_target': 10,  # The maximum R:R multiple to check for
}

def analyze_signals(cfg):
    """
    Main function to run the weekly signal analysis.
    """
    print("--- Starting Weekly Signal R:R Pattern Analysis ---")

    # --- 1. Load Entry Signals ---
    signals_filepath = os.path.join(cfg['log_folder'], cfg['scanner_strategy_name'], cfg['scanner_output_filename'])
    if not os.path.exists(signals_filepath):
        print(f"ERROR: Signals file not found at: {signals_filepath}")
        print("Please run the 'weekly_entry_scanner.py' script first.")
        return
        
    try:
        # Parse 'setup_week_date' and 'entry_date' as datetime objects
        signals_df = pd.read_csv(signals_filepath, parse_dates=['setup_week_date', 'entry_date'])
        print(f"Successfully loaded {len(signals_df)} entry signals.")
    except Exception as e:
        print(f"ERROR: Could not load signals file. Reason: {e}")
        return

    # --- 2. Load All Necessary Weekly Data ---
    print("Loading all required weekly historical data...")
    weekly_data = {}
    unique_symbols = signals_df['symbol'].unique()
    for symbol in unique_symbols:
        try:
            weekly_file = os.path.join(cfg['processed_data_folder'], f"{symbol}_weekly_with_indicators.csv")
            if os.path.exists(weekly_file):
                weekly_data[symbol] = pd.read_csv(weekly_file, index_col='datetime', parse_dates=True)
                # Ensure column names are consistent (lowercase, no dots)
                weekly_data[symbol].rename(columns=lambda x: x.lower().replace('.', '_'), inplace=True)
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
        entry_price = signal['trigger_price'] # Use trigger as the theoretical entry price
        setup_week_date = signal['setup_week_date']
        entry_date = signal['entry_date'] # The daily date the trigger was hit

        if symbol not in weekly_data:
            outcomes.append('Data Missing')
            continue

        df_weekly = weekly_data[symbol]

        # --- Calculate Initial Stop-Loss based on weekly candles ---
        try:
            # Find the weekly candle corresponding to the setup_week_date
            # Use 'ffill' to find the last valid weekly candle on or before the setup_week_date
            setup_week_loc_indexer = df_weekly.index.get_indexer([setup_week_date], method='ffill')
            if setup_week_loc_indexer[0] == -1:
                outcomes.append('SL Calc Error')
                continue
            setup_week_loc = setup_week_loc_indexer[0]
            
            # Slice weekly data for stop-loss lookback
            stop_loss_slice = df_weekly.iloc[max(0, setup_week_loc - cfg['stop_loss_lookback_weeks']) : setup_week_loc + 1] # Include setup week
            
            if stop_loss_slice.empty:
                outcomes.append('SL Calc Error - Empty Slice')
                continue
            
            stop_loss_price = stop_loss_slice['low'].min()
        except (KeyError, IndexError) as e:
            # print(f"SL Calc Error for {symbol} on {setup_week_date}: {e}") # Debugging
            outcomes.append('SL Calc Error')
            continue

        initial_risk = entry_price - stop_loss_price
        if initial_risk <= 0:
            outcomes.append('Invalid Risk')
            continue

        # --- Define R:R Targets ---
        rr_targets = {f'Hit {i}R': entry_price + (initial_risk * i) for i in range(1, cfg['max_rr_target'] + 1)}
        
        # --- Trace Weekly Price Action from the entry_date ---
        # Find the first weekly candle that starts on or after the entry_date
        trade_horizon_start_loc_indexer = df_weekly.index.get_indexer([entry_date], method='bfill')
        if trade_horizon_start_loc_indexer[0] == -1:
            outcomes.append('Trade Horizon Error')
            continue
        trade_horizon_start_loc = trade_horizon_start_loc_indexer[0]

        # Get the trade horizon in weekly candles
        trade_horizon = df_weekly.iloc[trade_horizon_start_loc : trade_horizon_start_loc + cfg['max_holding_weeks']]
        
        final_outcome = 'Hit SL' # Default outcome is failure
        highest_rr_achieved = 0

        for _, candle in trade_horizon.iterrows():
            if candle['low'] <= stop_loss_price:
                # If stop-loss hit, record the outcome and break
                break 

            for i in range(cfg['max_rr_target'], 0, -1):
                if i > highest_rr_achieved and candle['high'] >= rr_targets[f'Hit {i}R']:
                    highest_rr_achieved = i
                    break # Found the highest R:R for this candle, move to next candle
        
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
        # Count signals that achieved AT LEAST this R:R multiple
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
