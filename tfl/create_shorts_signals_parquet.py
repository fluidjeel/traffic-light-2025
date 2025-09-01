import pandas as pd
import os
import numpy as np
from pytz import timezone
import sys
import warnings

# SCRIPT VERSION v4.0 (Symmetrical Short Data Pipeline)
#
# ARCHITECTURAL NOTE:
# This script is a direct, symmetrical mirror of the hardened v4.0 long-side
# data pipeline. It is a complete replacement for all previous short data
# preparation scripts.
#
# KEY LOGIC (Inverted for Shorts):
# 1. Precisely isolates each "green candle series + red candle" pattern.
# 2. Calculates the TRUE lowest low and highest high for each specific pattern.
# 3. Applies ALL filters (RSI < 40, Trend < MA, Pattern Length) to the setup candle.
# 4. Verifies the breakdown on the subsequent candle against the true pattern low.

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION --
# ==============================================================================

# --- File Paths ---
ROOT_DIR = os.getcwd() 
DATA_DIRECTORY_15MIN = os.path.join(ROOT_DIR, "data", "universal_processed", "15min")
DATA_DIRECTORY_DAILY = os.path.join(ROOT_DIR, "data", "universal_processed", "daily")
SYMBOL_LIST_PATH = os.path.join(ROOT_DIR, "nifty200_fno.csv")
OUTPUT_DIRECTORY = os.path.join(ROOT_DIR, "data", "strategy_specific_data")
OUTPUT_FILENAME = "tfl_shorts_data_with_signals_v2.parquet"

# --- Additional Symbols ---
ADDITIONAL_SYMBOLS = ['NIFTY50-INDEX', 'NIFTYBANK-INDEX']

# --- Strategy Parameters (Symmetrical to Longs) ---
MIN_GREEN_CANDLES = 1
MAX_GREEN_CANDLES = 9
MVWAP_PERIOD = 50
INDEX_EMA_PERIOD = 50
RSI_THRESHOLD = 25.0 # Lower threshold for shorts
SLIPPAGE_PCT = 0.05 / 100

# ==============================================================================
# --- SIGNAL CALCULATION ENGINE ---
# ==============================================================================

def calculate_short_signals_for_symbol(symbol, symbol_daily_data):
    """
    Reads a symbol's 15min data, verifies it, calculates all short signals,
    and returns a DataFrame.
    """
    print(f"  - Processing: {symbol}", end='\r')
    tz = timezone('Asia/Kolkata')
    is_index = "-INDEX" in symbol

    data_path = os.path.join(DATA_DIRECTORY_15MIN, f"{symbol}_15min_with_indicators.parquet")
    if not os.path.exists(data_path): return None
    
    try:
        df = pd.read_parquet(data_path)
        if df.empty: return None
        
        df.reset_index(inplace=True)
        df.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.index = df.index.tz_localize(tz)

    except Exception as e:
        print(f"    > Error reading or cleaning data for {symbol}: {e}")
        return None

    if symbol_daily_data is None or symbol_daily_data.empty: return None
    daily_rsi_data = symbol_daily_data[['rsi_14']].copy()
    daily_rsi_data.rename(columns={'rsi_14': 'daily_rsi'}, inplace=True)
    df = pd.merge_asof(df.sort_index(), daily_rsi_data.sort_index(), left_index=True, right_index=True, direction='backward')

    is_red = df['close'] < df['open']
    is_green = df['close'] > df['open']
    
    # --- Accurate Pattern Detection (Inverted for Shorts) ---
    is_green_block_start = is_green & ~is_green.shift(1).fillna(False)
    green_block_id = is_green_block_start.cumsum()
    df['green_block_id'] = green_block_id.where(is_green)

    df['pattern_id'] = df['green_block_id'].shift(1).where(is_red & is_green.shift(1).fillna(False))
    
    df['pattern_group'] = df['green_block_id'].fillna(df['pattern_id'])
    
    valid_patterns = df[df['pattern_group'].notna()].copy()
    if not valid_patterns.empty:
        pattern_groups = valid_patterns.groupby('pattern_group')
        df['pattern_high'] = pattern_groups['high'].transform('max')
        df['pattern_low'] = pattern_groups['low'].transform('min')
        df['pattern_green_candle_count'] = pattern_groups['close'].transform(lambda x: (df.loc[x.index, 'close'] > df.loc[x.index, 'open']).sum())
    else:
        df['pattern_high'] = np.nan
        df['pattern_low'] = np.nan
        df['pattern_green_candle_count'] = 0

    # --- Apply ALL Filters to the Setup Candle (T-1, which is a red candle) ---
    rsi_filter_passed = df['daily_rsi'] < RSI_THRESHOLD
    if is_index:
        trend_filter_passed = df['close'] < df.get(f'ema_{INDEX_EMA_PERIOD}', np.nan)
    else:
        trend_filter_passed = df['close'] < df.get(f'mvwap_{MVWAP_PERIOD}', np.nan)
    
    pattern_length_filter = (df['pattern_green_candle_count'] >= MIN_GREEN_CANDLES) & (df['pattern_green_candle_count'] <= MAX_GREEN_CANDLES)
    is_valid_setup_candle = df['pattern_id'].notna() & rsi_filter_passed & trend_filter_passed & pattern_length_filter

    # --- Calculate Entry Signals & Prices based on Candle T ---
    is_fast_entry_trigger = is_valid_setup_candle.shift(1).fillna(False) & (df['low'] <= df['pattern_low'].shift(1))
    df['is_fast_entry'] = is_fast_entry_trigger
    base_fast_entry_price = np.minimum(df['pattern_low'].shift(1), df['open']) # Use minimum for shorts
    df['fast_entry_price'] = base_fast_entry_price * (1 - SLIPPAGE_PCT) # Adverse slippage is lower

    is_confirmation_candle = is_fast_entry_trigger
    df['is_confirmed_entry'] = is_confirmation_candle.shift(1).fillna(False)
    base_confirmed_entry_price = df['open']
    df['confirmed_entry_price'] = base_confirmed_entry_price * (1 - SLIPPAGE_PCT)
    
    df['symbol'] = symbol
    
    df['pattern_high'] = df['pattern_high'].where(is_valid_setup_candle)
    df['pattern_low'] = df['pattern_low'].where(is_valid_setup_candle)
    df['pattern_green_candle_count'] = df['pattern_green_candle_count'].where(is_valid_setup_candle)

    columns_to_keep = [
        'symbol', 'open', 'high', 'low', 'close', 'volume', 'daily_rsi',
        'is_fast_entry', 'fast_entry_price',
        'is_confirmed_entry', 'confirmed_entry_price',
        'pattern_high', 'pattern_low',
        'pattern_green_candle_count' 
    ]
    df_final = df[[col for col in columns_to_keep if col in df.columns]].copy()
    
    df_final = df_final[
        is_valid_setup_candle |
        df_final['is_fast_entry'].fillna(False) | 
        df_final['is_confirmed_entry'].fillna(False)
    ]

    return df_final

def main():
    print("--- Starting Shorts Strategy Data Pipeline (Symmetrical v4.0) ---")
    
    try:
        symbols_df = pd.read_csv(SYMBOL_LIST_PATH)
        symbols_to_process = symbols_df['symbol'].dropna().tolist()
        symbols_to_process.extend(ADDITIONAL_SYMBOLS)
        symbols_to_process = sorted(list(set(symbols_to_process)))
    except FileNotFoundError:
        print(f"ERROR: Symbol list not found at: {SYMBOL_LIST_PATH}. Exiting.")
        return
        
    print(f"Found {len(symbols_to_process)} symbols to process.")

    print("Pre-loading all daily data...")
    daily_data_map = {}
    tz = timezone('Asia/Kolkata')
    for symbol in symbols_to_process:
        daily_path = os.path.join(DATA_DIRECTORY_DAILY, f"{symbol}_daily_with_indicators.parquet")
        if os.path.exists(daily_path):
            df_daily = pd.read_parquet(daily_path)
            if not df_daily.empty:
                df_daily = df_daily[df_daily.index.notna()]
                if not df_daily.empty:
                    df_daily.index = df_daily.index.tz_localize(tz).normalize()
                    daily_data_map[symbol] = df_daily
    
    all_symbol_dfs = []
    for symbol in symbols_to_process:
        try:
            symbol_daily_data = daily_data_map.get(symbol)
            result_df = calculate_short_signals_for_symbol(symbol, symbol_daily_data)
            if result_df is not None and not result_df.empty:
                all_symbol_dfs.append(result_df)
        except Exception as e:
            print(f"\n---! An unexpected error occurred processing {symbol}: {e} !---")

    if not all_symbol_dfs:
        print("\n\nNo signal data was generated.")
        return

    print("\n\nCombining all processed symbols into master DataFrame...")
    master_df = pd.concat(all_symbol_dfs)
    master_df['symbol'] = master_df['symbol'].astype('category')
    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)
    
    print(f"Saving master data file to: {OUTPUT_FILENAME}")
    try:
        master_df.to_parquet(output_path)
        print("\n--- Data Pipeline Complete! ---")
        print(f"Successfully created '{OUTPUT_FILENAME}'.")
    except Exception as e:
        print(f"\nERROR: An error occurred during file save: {e}")

if __name__ == "__main__":
    main()
