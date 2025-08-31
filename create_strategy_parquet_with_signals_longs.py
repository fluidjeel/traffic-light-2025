import pandas as pd
import os
import numpy as np
from pytz import timezone
import sys
import warnings

# --- SUPPRESS FUTUREWARNING ---
# This line will suppress the specific FutureWarning from pandas related to downcasting.
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# --- File Paths ---
# The script now assumes it is being run from the project's root directory (e.g., D:\algo-2025).
ROOT_DIR = os.getcwd() # Use the current working directory as the project root.
DATA_DIRECTORY_15MIN = os.path.join(ROOT_DIR, "data", "universal_processed", "15min")
DATA_DIRECTORY_DAILY = os.path.join(ROOT_DIR, "data", "universal_processed", "daily")
# This path correctly points to nifty200_fno.csv in your project root directory.
SYMBOL_LIST_PATH = os.path.join(ROOT_DIR, "nifty200_fno.csv")
OUTPUT_DIRECTORY = os.path.join(ROOT_DIR, "data", "strategy_specific_data")
OUTPUT_FILENAME = "tfl_longs_data_with_signals.parquet"

# --- Additional Symbols ---
ADDITIONAL_SYMBOLS = ['NIFTY50-INDEX', 'NIFTYBANK-INDEX']

# --- Strategy Parameters ---
MIN_CONSECUTIVE_CANDLES = 1
MAX_PATTERN_LOOKBACK = 10
MVWAP_PERIOD = 50
INDEX_EMA_PERIOD = 50
RSI_THRESHOLD = 75.0
ATR_TS_PERIOD = 14
SLIPPAGE_PCT = 0.05 / 100 # 0.05% slippage

# ==============================================================================
# --- SIGNAL CALCULATION ENGINE ---
# ==============================================================================

def calculate_signals_for_symbol(symbol, symbol_daily_data):
    """
    Reads a symbol's 15min data, verifies it, and calculates both Fast and
    Confirmed entry signals.
    """
    print(f"  - Processing signals for: {symbol}")
    tz = timezone('Asia/Kolkata')
    is_index = "-INDEX" in symbol

    # --- 1. Load 15-Min Data ---
    data_path = os.path.join(DATA_DIRECTORY_15MIN, f"{symbol}_15min_with_indicators.parquet")
    if not os.path.exists(data_path):
        print(f"    > Warning: 15min data not found for {symbol}. Skipping.")
        return None
    
    try:
        df = pd.read_parquet(data_path)
        if df.empty:
            print(f"    > Warning: 15min data for {symbol} is empty. Skipping.")
            return None
        # --- 2. DATA VERIFICATION AND CLEANING ---
        # Ensure timezone localization first.
        df.index = pd.to_datetime(df.index).tz_localize(tz)
        
        # Check for and remove duplicate timestamps.
        if df.index.has_duplicates:
            print(f"    > Found and removed {df.index.duplicated().sum()} duplicate timestamps for {symbol}.")
            df = df[~df.index.duplicated(keep='last')]
            
        # Ensure data is sorted chronologically. This is CRITICAL for T+1 logic.
        if not df.index.is_monotonic_increasing:
            print(f"    > Data for {symbol} was not sorted. Sorting now.")
            df.sort_index(inplace=True)

    except Exception as e:
        print(f"    > Error reading or cleaning 15min data for {symbol}: {e}")
        return None

    # --- 3. Merge Daily Data ---
    if symbol_daily_data is None or symbol_daily_data.empty:
        print(f"    > Warning: Daily data not found for {symbol}. Cannot calculate signals.")
        return None
        
    daily_filters_to_map = symbol_daily_data[[f'atr_{ATR_TS_PERIOD}_pct', 'rsi_14']].copy()
    daily_filters_to_map.rename(columns={
        f'atr_{ATR_TS_PERIOD}_pct': 'daily_vol_pct',
        'rsi_14': 'daily_rsi'
    }, inplace=True)
    df = pd.merge_asof(df.sort_index(), daily_filters_to_map.sort_index(), left_index=True, right_index=True, direction='backward')

    # --- 4. Identify Base Pattern and Filters ---
    is_red = df['close'] < df['open']
    is_green = df['close'] > df['open']
    red_blocks = (is_red != is_red.shift()).cumsum()
    consecutive_reds = is_red.groupby(red_blocks).cumsum()
    consecutive_reds[~is_red] = 0

    num_red_candles_prev = consecutive_reds.shift(1).fillna(0)
    
    # This is the green "signal" candle (T-1 in our entry logic)
    is_signal_candle = (is_green & (num_red_candles_prev >= MIN_CONSECUTIVE_CANDLES) & (num_red_candles_prev <= MAX_PATTERN_LOOKBACK - 1))
    
    # Apply daily filters. This condition must be true on the signal candle.
    rsi_filter_passed = df['daily_rsi'] > RSI_THRESHOLD
    if is_index:
        trend_filter_passed = df['close'] > df.get(f'ema_{INDEX_EMA_PERIOD}')
    else:
        trend_filter_passed = df['close'] > df.get(f'mvwap_{MVWAP_PERIOD}')
    
    is_valid_setup = is_signal_candle & rsi_filter_passed & trend_filter_passed
    
    # --- 5. Calculate Trigger Price ---
    # Calculate the length of the pattern and the trigger price (highest high)
    df['pattern_len'] = np.where(is_signal_candle, num_red_candles_prev + 1, 0)
    
    # Forward fill the pattern length and trigger price to the next candle (the potential entry candle)
    df['ffill_pattern_len'] = df['pattern_len'].replace(0, np.nan).ffill()
    
    # Calculate the trigger price (highest high of the pattern)
    df['trigger_price'] = df['high'].rolling(window=df['ffill_pattern_len'].max().astype(int) if pd.notna(df['ffill_pattern_len'].max()) else 1).max().shift(1)
    df['trigger_price'] = np.where(df['pattern_len'] > 0, df['trigger_price'], np.nan)
    df['ffill_trigger_price'] = df['trigger_price'].ffill()

    # --- 6. Plot Entry Signals ---
    
    # A. Fast Entry
    is_confirmation_candle_T = is_valid_setup.shift(1).fillna(False)
    breakout_occurred = df['high'] > df['ffill_trigger_price']
    df['is_fast_entry'] = is_confirmation_candle_T & breakout_occurred
    df['fast_entry_price'] = df['ffill_trigger_price'] * (1 + SLIPPAGE_PCT)

    # B. Confirmed Entry
    # The confirmation candle T is the same as the fast entry candle
    is_confirmed_breakout = df['is_fast_entry'] 
    # The entry candle T+1 is the one AFTER the confirmed breakout
    df['is_confirmed_entry'] = is_confirmed_breakout.shift(1).fillna(False)
    df['confirmed_entry_price'] = df['open'].shift(-1) * (1 + SLIPPAGE_PCT) # Use next candle's open

    # --- 7. Finalize DataFrame ---
    df['symbol'] = symbol
    columns_to_keep = [
        'symbol', 'open', 'high', 'low', 'close',
        'is_fast_entry', 'fast_entry_price',
        'is_confirmed_entry', 'confirmed_entry_price',
        'daily_rsi'
    ]
    
    final_cols = [col for col in columns_to_keep if col in df.columns]
    
    # Set prices to NaN if the signal is False
    df.loc[~df['is_fast_entry'], 'fast_entry_price'] = np.nan
    df.loc[~df['is_confirmed_entry'], 'confirmed_entry_price'] = np.nan
    
    return df[final_cols]


def main():
    """Main function to orchestrate the signal data creation process."""
    print("--- Starting Long Signal Data Preparation (Fast & Confirmed Entries) ---")
    
    try:
        symbols_df = pd.read_csv(SYMBOL_LIST_PATH)
        symbols_from_csv = symbols_df['symbol'].tolist()
        symbols_to_process = sorted(list(set(symbols_from_csv + ADDITIONAL_SYMBOLS)))
    except FileNotFoundError:
        print(f"Warning: Symbol list not found at: {SYMBOL_LIST_PATH}. Running on indices only.")
        symbols_to_process = sorted(ADDITIONAL_SYMBOLS)
        
    print(f"Found {len(symbols_to_process)} total symbols to process.")

    print("Pre-loading all daily data for efficient merging...")
    daily_data_map = {}
    tz = timezone('Asia/Kolkata')
    for symbol in symbols_to_process:
        daily_path = os.path.join(DATA_DIRECTORY_DAILY, f"{symbol}_daily_with_indicators.parquet")
        if os.path.exists(daily_path):
            df_daily = pd.read_parquet(daily_path)
            df_daily = df_daily[df_daily.index.notna()]
            if not df_daily.empty:
                df_daily.index = df_daily.index.tz_localize(tz).normalize()
                daily_data_map[symbol] = df_daily
    
    # --- Sequential Execution ---
    all_symbol_dfs = []
    symbols_with_signals = {} # To store symbols that have signals for verification
    
    for symbol in symbols_to_process:
        try:
            symbol_daily_data = daily_data_map.get(symbol)
            result_df = calculate_signals_for_symbol(symbol, symbol_daily_data)
            if result_df is not None:
                all_symbol_dfs.append(result_df)
                # Check if signals were generated for verification later
                if result_df['is_fast_entry'].any():
                    symbols_with_signals['fast'] = symbol
                if result_df['is_confirmed_entry'].any():
                    symbols_with_signals['confirmed'] = symbol
        except Exception as e:
            print(f"---! An unexpected error occurred processing {symbol}: {e}, {sys.exc_info()[-1].tb_lineno} !---")


    if not all_symbol_dfs:
        print("\nNo data was processed. No output file will be created.")
        return

    print("\nCombining all processed symbols into a single master DataFrame...")
    master_df = pd.concat(all_symbol_dfs, ignore_index=False)
    
    master_df['symbol'] = master_df['symbol'].astype('category')
    
    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    print(f"Saving partitioned Parquet file to: {output_path}")
    master_df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        partition_cols=['symbol']
    )
    
    print("\n--- Process Complete! ---")
    
    # --- 8. FINAL VERIFICATION & SAMPLE PRINTING ---
    print("\n--- Verification Samples ---")
    for entry_type, symbol in symbols_with_signals.items():
        print(f"\nLoading sample for '{entry_type.title()} Entry' from symbol '{symbol}'...")
        try:
            # Read the specific partition for the symbol from the saved file
            symbol_df = pd.read_parquet(output_path, filters=[('symbol', '==', symbol)])
            signal_col = f'is_{entry_type}_entry'
            
            # FIX: Fill NA values with False to ensure a pure boolean mask.
            # This prevents the "Cannot mask with non-boolean array" error.
            sample = symbol_df[symbol_df[signal_col].fillna(False)].tail(5)
            
            if not sample.empty:
                print(sample)
            else:
                print(f"No '{entry_type}' signals found for symbol '{symbol}' in the final file.")

        except Exception as e:
            print(f"Could not load or display sample for {symbol}: {e}")


if __name__ == "__main__":
    main()
