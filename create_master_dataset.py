# create_master_dataset.py
#
# Description:
# This script consolidates all symbol data into a single, pre-processed Parquet file.
# It applies all trade entry filters (RSI, MVWAP/EMA, price action) and pre-computes
# a 'signal' column (1 for long, -1 for short, 0 for no signal) to create a master
# dataset for high-performance backtesting.
#
# INSTRUCTIONS:
# 1. Ensure you have the 'pyarrow' library installed: pip install pyarrow
# 2. Run this script once to generate the master file before running the simulator.
#
# This script separates the signal generation logic from the backtesting simulation
# to allow for faster, more flexible backtest optimizations.

import pandas as pd
import os
import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# The project root directory is determined automatically to build all other file paths.
# UPDATED: Hard-code the root directory to fix the file path issue.
ROOT_DIR = "D:\\algo-2025"

# --- LONGS CONFIGS ---
LONGS_MAX_PATTERN_LOOKBACK = 10
MIN_CONSECUTIVE_RED_CANDLES = 1
LONGS_RSI_THRESHOLD = 75.0
LONGS_MVWAP_PERIOD = 50

# --- SHORTS CONFIGS ---
SHORTS_MAX_PATTERN_LOOKBACK = 10
MIN_CONSECUTIVE_GREEN_CANDLES = 1
SHORTS_RSI_THRESHOLD = 25.0
SHORTS_MVWAP_PERIOD = 50

# Since indices lack volume for MVWAP, this EMA is used as the trend filter for them.
INDEX_EMA_PERIOD = 50

# --- DATA & FILE PATHS ---
DATA_DIRECTORY_15MIN = os.path.join(ROOT_DIR, "data", "universal_processed", "15min")
DATA_DIRECTORY_DAILY = os.path.join(ROOT_DIR, "data", "universal_processed", "daily")
SYMBOL_LIST_PATH = os.path.join(ROOT_DIR, "nifty200_fno.csv")
ADDITIONAL_SYMBOLS = ['NIFTY50-INDEX', 'NIFTYBANK-INDEX']
MASTER_FILE_PATH = os.path.join(ROOT_DIR, "data", "all_signals_master.parquet")

# ==============================================================================
# --- SIGNAL GENERATION ENGINE ---
# ==============================================================================

def precompute_signals_and_filters(df, daily_df, symbol):
    """
    Applies all entry pattern and confirmation filter logic to a single DataFrame.
    Returns a DataFrame with a new 'signal' column.
    This function is vectorized to be fast and lookahead-bias-free.
    """
    # Create the 'signal' column initialized to 0 (no signal)
    df['signal'] = 0

    if df.empty or daily_df.empty:
        return df

    is_index = "-INDEX" in symbol

    # --- Step 1: Pre-compute the price action pattern (vectorized) ---
    is_red = df['close'] < df['open']
    is_green = df['close'] > df['open']
    
    # Calculate consecutive red candles
    # NEW FIX: Correctly count consecutive candles
    red_blocks = (is_red != is_red.shift()).cumsum()
    consecutive_reds = is_red.groupby(red_blocks).cumsum()
    consecutive_reds[~is_red] = 0
    num_red_candles_prev = consecutive_reds.shift(1).fillna(0)
    
    # Calculate consecutive green candles
    # NEW FIX: Correctly count consecutive candles
    green_blocks = (is_green != is_green.shift()).cumsum()
    consecutive_greens = is_green.groupby(green_blocks).cumsum()
    consecutive_greens[~is_green] = 0
    num_green_candles_prev = consecutive_greens.shift(1).fillna(0)
    
    # --- Step 2: Determine if entry criteria are met for the next candle ---
    # Long signal: current candle is green and previous candles had a valid red pattern
    # CORRECTED LOGIC
    is_long_signal_pattern = (is_green & (num_red_candles_prev >= MIN_CONSECUTIVE_RED_CANDLES) & (num_red_candles_prev <= LONGS_MAX_PATTERN_LOOKBACK))
    
    # Short signal: current candle is red and previous candles had a valid green pattern
    # CORRECTED LOGIC
    is_short_signal_pattern = (is_red & (num_green_candles_prev >= MIN_CONSECUTIVE_GREEN_CANDLES) & (num_green_candles_prev <= SHORTS_MAX_PATTERN_LOOKBACK))

    # --- Step 3: Apply the confirmation filters ---
    # Use a date-based lookup to merge daily RSI data to the 15min data
    daily_df_temp = daily_df.rename(columns={'rsi_14': 'daily_rsi_14'})
    df = df.join(daily_df_temp['daily_rsi_14'], on=df.index.normalize())
    df['daily_rsi_14'].fillna(method='ffill', inplace=True)

    long_filters_met = (
        (df['daily_rsi_14'] >= LONGS_RSI_THRESHOLD) &
        ( (df['close'] > df[f'mvwap_{LONGS_MVWAP_PERIOD}']) | (is_index & (df['close'] > df[f'ema_{INDEX_EMA_PERIOD}'])) )
    )

    short_filters_met = (
        (df['daily_rsi_14'] <= SHORTS_RSI_THRESHOLD) &
        ( (df['close'] < df[f'mvwap_{SHORTS_MVWAP_PERIOD}']) | (is_index & (df['close'] < df[f'ema_{INDEX_EMA_PERIOD}'])) )
    )
    
    # --- Step 4: Combine pattern and filters into a final signal ---
    df.loc[is_long_signal_pattern & long_filters_met, 'signal'] = 1
    df.loc[is_short_signal_pattern & short_filters_met, 'signal'] = -1
    
    # Pre-calculate entry and stop loss for a potential trade (for the next candle)
    # This is also key for avoiding lookahead bias.
    df['entry_price'] = np.nan
    df['sl'] = np.nan
    
    # NEW AND MORE ROBUST FIX
    signal_entries = df[df['signal'] != 0].copy()
    for idx, row in signal_entries.iterrows():
        try:
            pattern_len = int(num_red_candles_prev.loc[idx]) if row['signal'] == 1 else int(num_green_candles_prev.loc[idx])
            pattern_start_idx = df.index.get_loc(idx) - pattern_len
            pattern_candles = df.iloc[pattern_start_idx : df.index.get_loc(idx) + 1]

            if row['signal'] == 1: # Long
                entry_price = pattern_candles['high'].max()
                sl = pattern_candles['low'].min()
            else: # Short
                entry_price = pattern_candles['low'].min()
                sl = pattern_candles['high'].max()
            
            df.loc[idx, 'entry_price'] = entry_price
            df.loc[idx, 'sl'] = sl
        except Exception as e:
            # Log the error for debugging and invalidate the signal
            print(f"[ERROR] Failed to calculate pattern for {symbol} at {idx}: {e}")
            df.loc[idx, 'signal'] = 0 
            df.loc[idx, 'entry_price'] = np.nan
            df.loc[idx, 'sl'] = np.nan
            
    # FIXED: Added 'daily_rsi_14' to the list of returned columns.
    return df[['open', 'high', 'low', 'close', 'volume', 'signal', 'entry_price', 'sl', 'daily_rsi_14']]


def process_symbol_data(symbol):
    """
    Worker function for parallel data loading and pre-computation.
    Loads, processes, and pre-computes signals for a single symbol.
    Returns a DataFrame with a new 'signal' column.
    """
    try:
        path_15min = os.path.join(DATA_DIRECTORY_15MIN, f"{symbol}_15min_with_indicators.parquet")
        path_daily = os.path.join(DATA_DIRECTORY_DAILY, f"{symbol}_daily_with_indicators.parquet")

        if not os.path.exists(path_15min) or not os.path.exists(path_daily):
            return None

        df15 = pd.read_parquet(path_15min)
        dfD = pd.read_parquet(path_daily)
        
        # Ensure indices are datetime objects for proper merging
        df15.index = pd.to_datetime(df15.index).tz_localize(None)
        dfD.index = pd.to_datetime(dfD.index).tz_localize(None)

        # Apply the signal logic
        df_processed = precompute_signals_and_filters(df15.copy(), dfD.copy(), symbol)
        
        # Reset the index and add a symbol column to the processed data before returning
        df_processed = df_processed.reset_index().rename(columns={'index': 'datetime'})
        df_processed.insert(1, 'symbol', symbol)

        # FIX: Explicitly convert 'datetime' column to a datetime object.
        df_processed['datetime'] = pd.to_datetime(df_processed['datetime'])

        # NEW FIX: Sort each DataFrame before it is returned from the parallel process.
        # This ensures that all chunks of data are in chronological order, making the
        # final concatenation and sorting more reliable.
        df_processed.sort_values(by='datetime', inplace=True)

        # Downcast data types to reduce memory usage
        for col in df_processed.columns:
            if df_processed[col].dtype == 'float64':
                df_processed[col] = pd.to_numeric(df_processed[col], downcast='float')
            if df_processed[col].dtype == 'int64':
                df_processed[col] = pd.to_numeric(df_processed[col], downcast='integer')

        return df_processed
    except Exception as e:
        print(f"[ERROR] Failed to process data for symbol {symbol}: {e}")
        return None

def main():
    """Main function to orchestrate the data consolidation process."""
    print("--- Starting Data Consolidation and Signal Pre-computation ---")
    
    try:
        symbols_df = pd.read_csv(SYMBOL_LIST_PATH)
        print("Column names in your CSV file are:", symbols_df.columns.tolist())
        symbols_to_process = symbols_df['symbol'].tolist()
    except FileNotFoundError:
        print(f"Warning: '{SYMBOL_LIST_PATH}' not found. Processing indices only.")
        symbols_to_process = []
    
    symbols_to_process.extend(ADDITIONAL_SYMBOLS)
    symbols_to_process = sorted(list(set(symbols_to_process)))

    print(f"\nFound {len(symbols_to_process)} symbols to process.")

    num_processes = max(1, cpu_count() - 1)
    print(f"Starting parallel processing with {num_processes} workers...")
    
    # ADDED: Print the configuration settings for easier debugging.
    print(f"  - Longs Max Pattern Lookback: {LONGS_MAX_PATTERN_LOOKBACK}")
    print(f"  - Shorts Min Consecutive Greens: {MIN_CONSECUTIVE_GREEN_CANDLES}")
    print(f"  - Longs RSI Threshold: {LONGS_RSI_THRESHOLD}")
    print(f"  - Shorts RSI Threshold: {SHORTS_RSI_THRESHOLD}")
    print(f"  - Longs MVWAP Period: {LONGS_MVWAP_PERIOD}")
    print(f"  - Shorts MVWAP Period: {SHORTS_MVWAP_PERIOD}")
    
    try:
        with Pool(processes=num_processes) as pool:
            all_dfs = [df for df in pool.map(process_symbol_data, symbols_to_process) if df is not None]
    except KeyboardInterrupt:
        print("\n--- Process interrupted by user. Terminating workers and exiting. ---")
        sys.exit(1)
        
    if not all_dfs:
        print("No valid data was found for any symbols. Exiting.")
        return
        
    master_df = pd.concat(all_dfs)
    master_df.sort_values(by=['datetime', 'symbol'], inplace=True)
    master_df.set_index('datetime', inplace=True)

    print(f"\nConsolidated data for {len(master_df['symbol'].unique())} symbols into a single DataFrame.")
    print(f"Saving master file to: {MASTER_FILE_PATH}")
    
    master_df.to_parquet(MASTER_FILE_PATH)

    print("\n--- Consolidation Complete! Master Parquet file is ready for backtesting. ---")

if __name__ == "__main__":
    main()
