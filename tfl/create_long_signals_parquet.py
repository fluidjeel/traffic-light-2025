import pandas as pd
import os
import numpy as np
from pytz import timezone
import sys
import warnings

# SCRIPT VERSION v4.0
#
# ARCHITECTURAL NOTE (Definitive Fix):
# This script is a complete re-engineering of the data pipeline, built to be a
# perfect, vectorized replication of the precise and correct logic found in the
# original, trusted `longs_simulator.py`. All previous flawed `groupby` logic
# has been abandoned.
#
# KEY LOGIC:
# 1. Precisely isolates each "red candle series + green candle" pattern.
# 2. Calculates the TRUE highest high and lowest low for each specific pattern.
# 3. Applies ALL filters (RSI, Trend, Pattern Length) to the setup candle.
# 4. Verifies the breakout on the subsequent candle against the true pattern high.
#
# This pipeline is now both fast and, most importantly, logically correct,
# resolving all previously identified signal generation flaws.

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
OUTPUT_FILENAME = "tfl_longs_data_with_signals_v2.parquet"

# --- Additional Symbols ---
ADDITIONAL_SYMBOLS = ['NIFTY50-INDEX', 'NIFTYBANK-INDEX']

# --- Strategy Parameters (from longs_simulator.py) ---
MIN_RED_CANDLES = 1
MAX_RED_CANDLES = 9
MVWAP_PERIOD = 50
INDEX_EMA_PERIOD = 50
RSI_THRESHOLD = 75.0
SLIPPAGE_PCT = 0.05 / 100

# ==============================================================================
# --- SIGNAL CALCULATION ENGINE ---
# ==============================================================================

def calculate_long_signals_for_symbol(symbol, symbol_daily_data):
    """
    Reads a symbol's 15min data, verifies it, calculates all long signals
    using the correct, precise pattern logic, and returns a DataFrame.
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
    
    # --- Accurate Pattern Detection (v4.0 Logic) ---
    
    # 1. Isolate each continuous block of red candles and give it a unique ID.
    is_red_block_start = is_red & ~is_red.shift(1).fillna(False)
    red_block_id = is_red_block_start.cumsum()
    df['red_block_id'] = red_block_id.where(is_red)

    # 2. Identify the Setup Candle (T-1): A green candle immediately following a red block.
    # The pattern_id is the ID of the red block that precedes it.
    df['pattern_id'] = df['red_block_id'].shift(1).where(is_green & is_red.shift(1).fillna(False))
    
    # 3. Create a unified group for each full pattern (red block + its green T-1 candle)
    df['pattern_group'] = df['red_block_id'].fillna(df['pattern_id'])
    
    # 4. Calculate the true dimensions for each unique pattern group.
    valid_patterns = df[df['pattern_group'].notna()].copy()
    if not valid_patterns.empty:
        pattern_groups = valid_patterns.groupby('pattern_group')
        df['pattern_high'] = pattern_groups['high'].transform('max')
        df['pattern_low'] = pattern_groups['low'].transform('min')
        df['pattern_red_candle_count'] = pattern_groups['close'].transform(lambda x: (df.loc[x.index, 'close'] < df.loc[x.index, 'open']).sum())
    else:
        df['pattern_high'] = np.nan
        df['pattern_low'] = np.nan
        df['pattern_red_candle_count'] = 0

    # 5. Apply ALL filters to the Setup Candle (T-1)
    rsi_filter_passed = df['daily_rsi'] > RSI_THRESHOLD
    if is_index:
        trend_filter_passed = df['close'] > df.get(f'ema_{INDEX_EMA_PERIOD}', np.nan)
    else:
        trend_filter_passed = df['close'] > df.get(f'mvwap_{MVWAP_PERIOD}', np.nan)
    
    pattern_length_filter = (df['pattern_red_candle_count'] >= MIN_RED_CANDLES) & (df['pattern_red_candle_count'] <= MAX_RED_CANDLES)
    is_valid_setup_candle = df['pattern_id'].notna() & rsi_filter_passed & trend_filter_passed & pattern_length_filter

    # --- Calculate Entry Signals & Prices based on Candle T ---
    
    # Fast Entry: Trigger on candle T, which follows a VALID setup candle T-1.
    is_fast_entry_trigger = is_valid_setup_candle.shift(1).fillna(False) & (df['high'] >= df['pattern_high'].shift(1))
    df['is_fast_entry'] = is_fast_entry_trigger
    base_fast_entry_price = np.maximum(df['pattern_high'].shift(1), df['open'])
    df['fast_entry_price'] = base_fast_entry_price * (1 + SLIPPAGE_PCT)

    # Confirmed Entry: Trigger on candle T+1
    is_confirmation_candle = is_fast_entry_trigger
    df['is_confirmed_entry'] = is_confirmation_candle.shift(1).fillna(False)
    base_confirmed_entry_price = df['open']
    df['confirmed_entry_price'] = base_confirmed_entry_price * (1 + SLIPPAGE_PCT)
    
    df['symbol'] = symbol
    
    # Only keep the values on the candle where they are relevant to avoid confusion.
    df['pattern_high'] = df['pattern_high'].where(is_valid_setup_candle)
    df['pattern_low'] = df['pattern_low'].where(is_valid_setup_candle)
    df['pattern_red_candle_count'] = df['pattern_red_candle_count'].where(is_valid_setup_candle)

    columns_to_keep = [
        'symbol', 'open', 'high', 'low', 'close', 'volume', 'daily_rsi',
        'is_fast_entry', 'fast_entry_price',
        'is_confirmed_entry', 'confirmed_entry_price',
        'pattern_high', 'pattern_low',
        'pattern_red_candle_count' 
    ]
    df_final = df[[col for col in columns_to_keep if col in df.columns]].copy()
    
    # Keep only rows that are either a setup or an entry signal to keep the file lean.
    df_final = df_final[
        is_valid_setup_candle |
        df_final['is_fast_entry'].fillna(False) | 
        df_final['is_confirmed_entry'].fillna(False)
    ]

    return df_final

# ==============================================================================
# --- MAIN ORCHESTRATOR ---
# ==============================================================================

def main():
    """Main function to run the entire data pipeline."""
    print("--- Starting Longs Strategy Data Pipeline (v4.0 - Corrected Logic) ---")
    
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
            result_df = calculate_long_signals_for_symbol(symbol, symbol_daily_data)
            if result_df is not None and not result_df.empty:
                all_symbol_dfs.append(result_df)
        except Exception as e:
            print(f"\n---! An unexpected error occurred processing {symbol}: {e} !---")

    if not all_symbol_dfs:
        print("\n\nNo signal data was generated. This may be expected if the strategy rules are very strict.")
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

