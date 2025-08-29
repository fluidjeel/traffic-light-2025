# create_strategy_parquet_fast_entry.py
#
# Description:
# This script prepares the data for the "Fast Entry" simulator. It separates the
# "Alert Candle" (where conditions are met) from the "Entry Signal Candle"
# (where the price trigger is breached).
#
# FIX:
# - Corrected a KeyError by ensuring the 'datetime' column is correctly saved
#   to the final Parquet file.

import os
import pandas as pd
import numpy as np
from pytz import timezone
import warnings
import shutil

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIRECTORY_15MIN = os.path.join(ROOT_DIR, "data", "universal_processed", "15min")
DATA_DIRECTORY_DAILY = os.path.join(ROOT_DIR, "data", "universal_processed", "daily")
SYMBOL_LIST_PATH = os.path.join(ROOT_DIR, "nifty200_fno.csv")
ADDITIONAL_SYMBOLS = ['NIFTY50-INDEX', 'NIFTYBANK-INDEX']
OUTPUT_DIRECTORY = os.path.join(ROOT_DIR, "data", "strategy_specific_data")
OUTPUT_PARQUET_FILENAME = "tfl_shorts_data_fast_entry.parquet" # New data file

# --- STRATEGY PARAMETERS ---
MAX_PATTERN_LOOKBACK = 10
MIN_CONSECUTIVE_CANDLES = 1
RSI_THRESHOLD = 25.0
MVWAP_PERIOD = 50
INDEX_EMA_PERIOD = 50

REQUIRED_15MIN_COLS = [
    'open', 'high', 'low', 'close', 'atr_14',
    f'mvwap_{MVWAP_PERIOD}', f'ema_{INDEX_EMA_PERIOD}', 'volume_ratio'
]
REQUIRED_DAILY_COLS = ['rsi_14', 'atr_14_pct']

def calculate_fast_entry_signals(df, symbol):
    """
    Calculates alert candles (T-1) and entry signals (T).
    """
    if df.empty:
        return df

    # --- 1. Identify the Alert Candle (T-1) ---
    is_red = df['close'] < df['open']
    is_green = df['close'] > df['open']
    green_blocks = (is_green != is_green.shift()).cumsum()
    consecutive_greens = is_green.groupby(green_blocks).cumsum()
    consecutive_greens[~is_green] = 0
    num_green_candles_prev = consecutive_greens.shift(1).fillna(0)

    is_pattern_candle = (
        is_red &
        (num_green_candles_prev >= MIN_CONSECUTIVE_CANDLES) &
        (num_green_candles_prev <= MAX_PATTERN_LOOKBACK - 1)
    )

    rsi_filter_passed = df['daily_rsi'] < RSI_THRESHOLD
    is_index = "-INDEX" in symbol
    trend_filter_passed = df['close'] < df[f'ema_{INDEX_EMA_PERIOD}'] if is_index else df['close'] < df[f'mvwap_{MVWAP_PERIOD}']
    
    df['is_alert_candle'] = is_pattern_candle & rsi_filter_passed & trend_filter_passed

    # --- 2. Identify the Entry Signal Candle (T) ---
    df['alert_candle_low'] = df['low'].where(df['is_alert_candle']).shift(1)
    df['pattern_len'] = (num_green_candles_prev + 1).where(df['is_alert_candle']).shift(1)
    
    highs = df['high'].to_numpy()
    pattern_highs = np.full(len(df), np.nan)
    indices = np.arange(len(df))
    
    trigger_indices = np.where(df['alert_candle_low'].notna())[0]

    for i in trigger_indices:
        pattern_len = df['pattern_len'].iloc[i]
        if pd.notna(pattern_len):
            start_idx = max(0, i - 1 - int(pattern_len) + 1)
            pattern_highs[i] = np.max(highs[start_idx : i])

    df['pattern_high_for_sl'] = pattern_highs
    
    df['is_entry_signal'] = df['low'] < df['alert_candle_low']
    
    return df


def main():
    """Main function to orchestrate the data consolidation and signal calculation."""
    print("--- Starting Parquet File Creation for FAST ENTRY Signals ---")

    try:
        symbols_df = pd.read_csv(SYMBOL_LIST_PATH)
        symbols_to_process = symbols_df['symbol'].tolist() + ADDITIONAL_SYMBOLS
        symbols_to_process = sorted(list(set(symbols_to_process)))
    except FileNotFoundError:
        print(f"ERROR: Symbol list not found at {SYMBOL_LIST_PATH}. Exiting.")
        return

    all_symbol_dfs = []
    tz = timezone('Asia/Kolkata')

    for symbol in symbols_to_process:
        try:
            path_15min = os.path.join(DATA_DIRECTORY_15MIN, f"{symbol}_15min_with_indicators.parquet")
            path_daily = os.path.join(DATA_DIRECTORY_DAILY, f"{symbol}_daily_with_indicators.parquet")

            if not os.path.exists(path_15min) or not os.path.exists(path_daily): continue

            df_15min = pd.read_parquet(path_15min)
            df_daily = pd.read_parquet(path_daily)

            if df_15min.empty or df_daily.empty: continue

            df_15min = df_15min[REQUIRED_15MIN_COLS].copy()
            df_daily = df_daily[REQUIRED_DAILY_COLS].copy()
            df_daily = df_daily[df_daily.index.notna()]

            df_15min.index = pd.to_datetime(df_15min.index).tz_localize(tz)
            df_daily.index = pd.to_datetime(df_daily.index).tz_localize(tz).normalize()

            df_daily.rename(columns={'rsi_14': 'daily_rsi', 'atr_14_pct': 'daily_vol_pct'}, inplace=True)

            df_merged = pd.merge_asof(
                df_15min.sort_index(), df_daily.sort_index(),
                left_index=True, right_index=True, direction='backward'
            )
            df_merged.dropna(inplace=True)

            df_with_signals = calculate_fast_entry_signals(df_merged, symbol)
            df_with_signals['symbol'] = symbol
            all_symbol_dfs.append(df_with_signals)

        except Exception as e:
            print(f"    - ERROR processing {symbol}: {e}")

    if not all_symbol_dfs:
        print("No data was processed. Exiting.")
        return

    final_df = pd.concat(all_symbol_dfs, ignore_index=False)
    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_PARQUET_FILENAME)
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    print(f"Saving partitioned Parquet file to: {output_path}")

    # FIX: Add 'datetime' to the list of columns to keep
    final_cols_to_keep = [
        'datetime', 'open', 'high', 'low', 'close', 'atr_14', 'daily_rsi', 'daily_vol_pct',
        'is_alert_candle', 'is_entry_signal', 'alert_candle_low', 'pattern_high_for_sl',
        'volume_ratio', 'symbol'
    ]
    final_df = final_df.reset_index().rename(columns={'index': 'datetime'})
    final_cols_to_keep = [col for col in final_cols_to_keep if col in final_df.columns]
    final_df = final_df[final_cols_to_keep]

    final_df.to_parquet(path=output_path, engine='pyarrow', partition_cols=['symbol'])

    print("\n--- Process Complete! ---")

if __name__ == "__main__":
    main()
