# create_strategy_parquet_with_signals.py
#
# Description:
# This script consolidates all necessary 15-minute and daily data for the
# TrafficLight-Manny (Shorts) strategy into a single, optimized, and partitioned
# Parquet file.
#
# ENHANCEMENT:
# This version now pre-calculates and flags all potential short entry signals
# based on the exact logic from shorts_simulator.py. It adds a new boolean
# column 'is_entry_signal' to the final dataset, allowing for easy plotting
# and analysis while being 100% free of lookahead bias.
# It also includes a final verification and sampling step to confirm data
# integrity and signal accuracy.
#
# FIX:
# - Corrected a KeyError by ensuring the 'symbol' column is kept for partitioning.
# - Added dropna() after merging dataframes to remove rows with null values
#   that were causing the verification step to fail.

import os
import pandas as pd
import numpy as np
from pytz import timezone
import warnings
import shutil

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION (Should match shorts_simulator.py and project structure) ---
# ==============================================================================

# -- PROJECT ROOT DIRECTORY --
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.abspath(__file__)))

# -- SOURCE DATA DIRECTORIES --
DATA_DIRECTORY_15MIN = os.path.join(ROOT_DIR, "data", "universal_processed", "15min")
DATA_DIRECTORY_DAILY = os.path.join(ROOT_DIR, "data", "universal_processed", "daily")

# -- SYMBOL LIST --
SYMBOL_LIST_PATH = os.path.join(ROOT_DIR, "nifty200_fno.csv")
ADDITIONAL_SYMBOLS = ['NIFTY50-INDEX', 'NIFTYBANK-INDEX']

# -- OUTPUT CONFIGURATION --
OUTPUT_DIRECTORY = os.path.join(ROOT_DIR, "data", "strategy_specific_data")
OUTPUT_PARQUET_FILENAME = "tfl_shorts_data_with_signals.parquet"

# --- STRATEGY PARAMETERS (from shorts_simulator.py) ---
MAX_PATTERN_LOOKBACK = 10
MIN_CONSECUTIVE_CANDLES = 1
RSI_THRESHOLD = 25.0
MVWAP_PERIOD = 50
INDEX_EMA_PERIOD = 50

# -- STRATEGY-SPECIFIC INDICATORS --
REQUIRED_15MIN_COLS = [
    'open', 'high', 'low', 'close',
    'atr_14',
    f'mvwap_{MVWAP_PERIOD}',
    f'ema_{INDEX_EMA_PERIOD}'
]
REQUIRED_DAILY_COLS = ['rsi_14', 'atr_14_pct']


def calculate_entry_signals(df, symbol):
    """
    Calculates all potential short entry signals using vectorized operations
    to ensure no lookahead bias.
    """
    if df.empty:
        return df

    # --- 1. Price Action Pattern Detection ---
    is_red = df['close'] < df['open']
    is_green = df['close'] > df['open']
    green_blocks = (is_green != is_green.shift()).cumsum()
    consecutive_greens = is_green.groupby(green_blocks).cumsum()
    consecutive_greens[~is_green] = 0
    num_green_candles_prev = consecutive_greens.shift(1).fillna(0)

    # A "signal candle" is a red candle preceded by 1 to 9 green candles
    is_signal_candle = (
        is_red &
        (num_green_candles_prev >= MIN_CONSECUTIVE_CANDLES) &
        (num_green_candles_prev <= MAX_PATTERN_LOOKBACK - 1)
    )

    # --- 2. Trend and Momentum Filter Conditions ---
    # Daily RSI filter (already merged as 'daily_rsi')
    rsi_filter_passed = df['daily_rsi'] < RSI_THRESHOLD

    # Intraday trend filter (adaptive for stocks vs. indices)
    is_index = "-INDEX" in symbol
    if is_index:
        trend_filter_passed = df['close'] < df[f'ema_{INDEX_EMA_PERIOD}']
    else:
        trend_filter_passed = df['close'] < df[f'mvwap_{MVWAP_PERIOD}']

    # Combine all filter conditions
    all_filters_passed = rsi_filter_passed & trend_filter_passed

    # --- 3. Identify Valid Setups ---
    # A valid setup exists on the signal candle if all filters passed
    valid_setup_candle = is_signal_candle & all_filters_passed

    # --- 4. Determine Entry Trigger ---
    # To avoid lookahead bias, the signal is shifted to the *next* candle.
    # This 'trade_trigger' column marks the candle where we check for entry.
    df['trade_trigger'] = valid_setup_candle.shift(1).fillna(False)

    # Calculate the pattern length on the signal candle and shift it
    df['pattern_len'] = (num_green_candles_prev + 1).shift(1).fillna(0)

    # Find the pattern's low to use as the entry trigger price
    # Note: This is a complex rolling operation, so we do it carefully.
    # We create an array of indices to define the start of each pattern.
    indices = np.arange(len(df))
    pattern_len_arr = df['pattern_len'].to_numpy(dtype=int)
    # The signal candle is at index i-1, pattern starts at i-1-pattern_len+1 = i-pattern_len
    pattern_start_indices = np.maximum(0, indices - pattern_len_arr)

    # This is a memory-intensive but accurate way to do a dynamic rolling window
    lows = df['low'].to_numpy()
    pattern_lows = np.full(len(df), np.nan)

    # Iterate where a trigger is possible to calculate the dynamic rolling low
    for i in np.where(df['trade_trigger'])[0]:
         start_idx = pattern_start_indices[i]
         # The pattern is from the start index up to the signal candle (i-1)
         pattern_lows[i] = np.min(lows[start_idx : i])

    df['pattern_low_trigger'] = pattern_lows

    # --- 5. Final Entry Signal ---
    # The final entry signal is True if a trade is triggered AND the current
    # candle's low breaks below the pattern's low.
    df['is_entry_signal'] = df['trade_trigger'] & (df['low'] < df['pattern_low_trigger'])

    return df


def verify_and_sample_data(parquet_path):
    """
    Loads the newly created Parquet file, runs verification checks,
    and prints a sample of entry signals for manual validation.
    """
    print("\n--- Starting Verification and Sampling ---")
    if not os.path.exists(parquet_path):
        print(f"ERROR: Parquet file not found at {parquet_path}. Verification failed.")
        return

    try:
        df = pd.read_parquet(parquet_path)
        print("✅ Successfully loaded the Parquet file.")

        # --- Verification Checks ---
        if df.empty:
            print("❌ VERIFICATION FAILED: The DataFrame is empty.")
            return

        required_cols = {'symbol', 'is_entry_signal', 'daily_rsi', 'close'}
        if not required_cols.issubset(df.columns):
            print(f"❌ VERIFICATION FAILED: Missing one or more required columns. Found: {list(df.columns)}")
            return
        print("✅ All required columns are present.")

        if df['close'].isnull().any() or df['daily_rsi'].isnull().any():
            print("❌ VERIFICATION FAILED: Found Null values in critical 'close' or 'daily_rsi' columns.")
            return
        print("✅ No null values found in critical columns.")

        # --- Signal Sampling ---
        signals_df = df[df['is_entry_signal']].copy()
        total_signals = len(signals_df)
        print(f"\nFound a total of {total_signals} entry signals across all symbols.")

        if total_signals > 0:
            num_samples = min(5, total_signals)
            print(f"--- Displaying {num_samples} Random Signal Samples for Manual Verification ---")
            
            for _, signal in signals_df.sample(n=num_samples).iterrows():
                symbol = signal['symbol']
                is_index = "-INDEX" in symbol
                trend_col = f'ema_{INDEX_EMA_PERIOD}' if is_index else f'mvwap_{MVWAP_PERIOD}'
                trend_val = signal.get(trend_col, 'N/A')

                print("-" * 50)
                print(f"  Symbol:    {symbol}")
                print(f"  Timestamp: {signal['datetime']}")
                print(f"  Condition 1 (Price < Trend):")
                print(f"    - Close Price: {signal['close']:.2f}")
                print(f"    - Trend ({trend_col}): {trend_val:.2f}")
                print(f"    - Met: {signal['close'] < trend_val}")
                print(f"  Condition 2 (Daily RSI < Threshold):")
                print(f"    - Daily RSI: {signal['daily_rsi']:.2f}")
                print(f"    - Threshold: < {RSI_THRESHOLD}")
                print(f"    - Met: {signal['daily_rsi'] < RSI_THRESHOLD}")
            print("-" * 50)
        else:
            print("No entry signals were found with the current parameters.")

    except Exception as e:
        print(f"❌ An error occurred during verification: {e}")


def main():
    """Main function to orchestrate the data consolidation and signal calculation."""
    print("--- Starting Enhanced Parquet File Creation with Entry Signals ---")

    # --- 1. Get list of all symbols ---
    try:
        symbols_df = pd.read_csv(SYMBOL_LIST_PATH)
        symbols_to_process = symbols_df['symbol'].tolist()
        symbols_to_process.extend(ADDITIONAL_SYMBOLS)
        symbols_to_process = sorted(list(set(symbols_to_process)))
        print(f"Found {len(symbols_to_process)} symbols to process.")
    except FileNotFoundError:
        print(f"ERROR: Symbol list not found at {SYMBOL_LIST_PATH}. Exiting.")
        return

    # --- 2. Process each symbol and collect dataframes ---
    all_symbol_dfs = []
    tz = timezone('Asia/Kolkata')

    for symbol in symbols_to_process:
        print(f"  - Processing {symbol}...")
        try:
            # --- a. Load data ---
            path_15min = os.path.join(DATA_DIRECTORY_15MIN, f"{symbol}_15min_with_indicators.parquet")
            path_daily = os.path.join(DATA_DIRECTORY_DAILY, f"{symbol}_daily_with_indicators.parquet")

            if not os.path.exists(path_15min) or not os.path.exists(path_daily):
                print(f"    - Warning: Data file missing for {symbol}. Skipping.")
                continue

            df_15min = pd.read_parquet(path_15min)
            df_daily = pd.read_parquet(path_daily)

            if df_15min.empty or df_daily.empty:
                print(f"    - Warning: Data file for {symbol} is empty. Skipping.")
                continue

            # --- b. Clean, select, and merge data ---
            df_15min = df_15min[REQUIRED_15MIN_COLS].copy()
            df_daily = df_daily[REQUIRED_DAILY_COLS].copy()
            df_daily = df_daily[df_daily.index.notna()]

            df_15min.index = df_15min.index.tz_localize(tz)
            df_daily.index = df_daily.index.tz_localize(tz).normalize()

            df_daily.rename(columns={'rsi_14': 'daily_rsi', 'atr_14_pct': 'daily_vol_pct'}, inplace=True)

            df_merged = pd.merge_asof(
                df_15min.sort_index(), df_daily.sort_index(),
                left_index=True, right_index=True, direction='backward'
            )
            
            # *** FIX: Drop rows with nulls after merging to prevent verification error. ***
            df_merged.dropna(inplace=True)

            # --- c. Calculate entry signals ---
            df_with_signals = calculate_entry_signals(df_merged, symbol)
            df_with_signals['symbol'] = symbol
            all_symbol_dfs.append(df_with_signals)

        except Exception as e:
            print(f"    - ERROR processing {symbol}: {e}")

    if not all_symbol_dfs:
        print("No data was processed. Exiting.")
        return

    # --- 3. Concatenate and save final dataset ---
    print("\nConcatenating data for all symbols...")
    final_df = pd.concat(all_symbol_dfs, ignore_index=False)

    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_PARQUET_FILENAME)
    if os.path.exists(output_path):
        print(f"Removing existing directory: {output_path}")
        shutil.rmtree(output_path)

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"Saving partitioned Parquet file to: {output_path}")

    # Select final columns to keep the file clean.
    # *** FIX: Added 'symbol' to this list to ensure it's available for partitioning. ***
    final_cols_to_keep = [
        'open', 'high', 'low', 'close', 'atr_14', f'mvwap_{MVWAP_PERIOD}',
        f'ema_{INDEX_EMA_PERIOD}', 'daily_rsi', 'daily_vol_pct', 'is_entry_signal',
        'symbol'
    ]
    # Filter the DataFrame, keeping only the essential columns plus the 'symbol' for partitioning.
    # The helper columns from signal calculation are now dropped.
    final_df = final_df[final_cols_to_keep]

    # Set datetime as a column before saving for easier loading
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'datetime'}, inplace=True)

    final_df.to_parquet(
        path=output_path,
        engine='pyarrow',
        partition_cols=['symbol']
    )

    print("\n--- Process Complete! ---")
    print("Your strategy-specific Parquet file with pre-calculated entry signals is ready.")

    # --- 4. NEW: Verify the created file and sample signals ---
    verify_and_sample_data(output_path)


if __name__ == "__main__":
    main()
