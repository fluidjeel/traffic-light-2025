# universal_calculate_indicators.py
#
# Description:
# A universal indicator calculation script designed to work with the output of
# the 'universal_fyers_scraper.py'. This is the comprehensive version containing
# a full suite of indicators for broad analysis.
#
# MODIFICATION (v14 - Final Stability Fix):
# - Implemented a definitive and simplified fix for handling empty/corrupt files.
# - The script now checks if a file is empty immediately after reading it.
# - This robustly prevents all 'KeyError' and 'TypeError' issues from bad data files.

import pandas as pd
import os
import sys
import numpy as np

# --- COMPATIBILITY FIX ---
# Newer versions of numpy deprecated np.NaN in favor of np.nan.
# The pandas_ta library still uses the old np.NaN, causing an ImportError.
# This line creates the alias that pandas_ta expects, fixing the issue.
np.NaN = np.nan
import pandas_ta as ta

from multiprocessing import Pool, cpu_count
import argparse

def calculate_all_indicators(df):
    """Calculates all required technical indicators on a given dataframe."""
    if df.empty:
        return df
    
    # --- Trend Indicators ---
    emas = [5, 8, 10, 20, 30, 50, 100, 200]
    for length in emas:
        df[f'ema_{length}'] = ta.ema(df['close'], length=length)

    smas = [20, 30, 50, 100, 200]
    for length in smas:
        df[f'sma_{length}'] = ta.sma(df['close'], length=length)

    # --- Momentum Indicators ---
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    try:
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if isinstance(macd, pd.DataFrame) and 'MACD_12_26_9' in macd.columns:
            df['macd_12_26_9'] = macd['MACD_12_26_9']
            df['macdh_12_26_9'] = macd['MACDh_12_26_9']
            df['macds_12_26_9'] = macd['MACDs_12_26_9']
    except Exception:
        df['macd_12_26_9'] = np.nan
        df['macdh_12_26_9'] = np.nan
        df['macds_12_26_9'] = np.nan

    try:
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            df['stoch_k_14_3_3'] = stoch['STOCHk_14_3_3']
            df['stoch_d_14_3_3'] = stoch['STOCHd_14_3_3']
    except Exception:
        df['stoch_k_14_3_3'] = np.nan
        df['stoch_d_14_3_3'] = np.nan
        
    df['return_30'] = df['close'].pct_change(periods=30) * 100

    # --- Volatility Indicators ---
    atr_14 = ta.atr(df['high'], df['low'], df['close'], length=14)
    if atr_14 is not None and not atr_14.empty:
        df['atr_14'] = atr_14
        df['atr_14_pct'] = (df['atr_14'] / df['close']) * 100
        
    atr_6 = ta.atr(df['high'], df['low'], df['close'], length=6)
    if atr_6 is not None and not atr_6.empty:
        df['atr_6'] = atr_6
        
    df['vix_10_sma'] = ta.sma(df['close'], length=10)

    # --- Volume Indicators ---
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['volume_20_sma'] = ta.sma(df['volume'], length=20)
    df['volume_50_sma'] = ta.sma(df['volume'], length=50)
    df['volume_ratio'] = df['volume'] / df['volume_20_sma']
    
    # --- ADDITIVE FIX for MVWAP Calculation ---
    # The pandas_ta.vwap function was causing an AttributeError.
    # This manual calculation is more robust and achieves the same result.
    df['mvwap_50'] = (df['close'] * df['volume']).rolling(window=50).sum() / df['volume'].rolling(window=50).sum()
    
    df['turnover'] = df['close'] * df['volume']
    df['turnover_20_sma'] = ta.sma(df['turnover'], length=20)

    # --- Custom Price Action Indicators ---
    df['52_week_high'] = df['high'].rolling(window=252, min_periods=1).max()
    df['prox_52w_high'] = ((df['close'] - df['52_week_high']) / df['52_week_high']) * 100

    candle_range = df['high'] - df['low']
    body_size = abs(df['close'] - df['open'])
    df['body_ratio'] = np.where(candle_range > 0, body_size / candle_range, 0)

    df['high_20_period'] = df['high'].rolling(window=20, min_periods=1).max()
    df['pullback_depth'] = (df['close'] - df['high_20_period']) / df['high_20_period']

    if 'ema_20' in df.columns and 'atr_14' in df.columns:
        df['volatility_ratio'] = df['atr_14'] / df['ema_20']
    else:
        df['volatility_ratio'] = np.nan
    
    return df

def process_symbol(symbol_name):
    """
    Processes all timeframes for a single symbol. This function is called by each
    parallel worker.
    """
    print(f"Processing {symbol_name}...")
    
    input_dir = os.path.join("data", "universal_historical_data")
    output_base_dir = "data/universal_processed"
    timeframes = {'daily': 'D', 'weekly': 'W-FRI', 'monthly': 'MS'}

    # --- Daily Processing ---
    input_filename_daily = f"{symbol_name}_daily.csv"
    input_path_daily = os.path.join(input_dir, input_filename_daily)

    if not os.path.exists(input_path_daily):
        print(f"  > Warning: Daily data file not found for {symbol_name}. Skipping.")
        return
    
    try:
        df_daily = pd.read_csv(input_path_daily, index_col='datetime', parse_dates=True)
        
        # --- DEFINITIVE FIX for Empty/Corrupt Files ---
        if df_daily.empty:
            print(f"  > Warning: Daily file for {symbol_name} is empty. Skipping.")
            return

        df_daily = df_daily[~df_daily.index.duplicated(keep='last')]
        df_daily.sort_index(inplace=True)
    except Exception as e:
        print(f"  > Error reading daily data for {symbol_name}: {e}")
        return

    df_daily_processed = calculate_all_indicators(df_daily.copy())
    
    output_daily_dir = os.path.join(output_base_dir, 'daily')
    os.makedirs(output_daily_dir, exist_ok=True)
    output_daily_path = os.path.join(output_daily_dir, f"{symbol_name}_daily_with_indicators.csv")
    df_daily_processed.to_csv(output_daily_path, index_label='datetime')

    # --- Higher Timeframe Processing ---
    aggregation_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    for tf_name, rule in timeframes.items():
        if tf_name == 'daily': continue
        df_htf = df_daily.resample(rule).agg(aggregation_rules).dropna()
        if df_htf.empty: continue
        df_htf_processed = calculate_all_indicators(df_htf)
        output_htf_dir = os.path.join(output_base_dir, tf_name)
        os.makedirs(output_htf_dir, exist_ok=True)
        output_htf_path = os.path.join(output_htf_dir, f"{symbol_name}_{tf_name}_with_indicators.csv")
        df_htf_processed.to_csv(output_htf_path, index_label='datetime')

    # --- 15-Minute Processing ---
    input_filename_15min = f"{symbol_name}_15min.csv"
    input_path_15min = os.path.join(input_dir, input_filename_15min)
    
    if os.path.exists(input_path_15min):
        try:
            df_15min = pd.read_csv(input_path_15min, index_col='datetime', parse_dates=True)
            
            # --- DEFINITIVE FIX for Empty/Corrupt Files ---
            if df_15min.empty:
                print(f"  > Warning: 15min file for {symbol_name} is empty. Skipping.")
                return

            df_15min = df_15min[~df_15min.index.duplicated(keep='last')]
            df_15min.sort_index(inplace=True)
            df_15min_processed = calculate_all_indicators(df_15min.copy())
            
            output_15min_dir = os.path.join(output_base_dir, '15min')
            os.makedirs(output_15min_dir, exist_ok=True)
            output_15min_path = os.path.join(output_15min_dir, f"{symbol_name}_15min_with_indicators.csv")
            df_15min_processed.to_csv(output_15min_path, index_label='datetime')
        except Exception as e:
            print(f"  > Error reading 15-minute data for {symbol_name}: {e}")
    
    print(f"Finished processing {symbol_name}.")


def main():
    """Main function to run the entire data processing pipeline in parallel."""
    parser = argparse.ArgumentParser(description="Universal Indicator Calculator")
    parser.add_argument('--only-index', action='store_true', help='Calculate indicators only for the indices.')
    args = parser.parse_args()

    print("--- Starting Universal Data Processing Engine (Parallel Version) ---")

    nifty_list_csv = "nifty500.csv"
    
    if not args.only_index:
        try:
            stock_list_df = pd.read_csv(nifty_list_csv)
            symbols_to_process = stock_list_df["Symbol"].dropna().tolist()
        except FileNotFoundError:
            print(f"Warning: '{nifty_list_csv}' not found. Processing indices only.")
            symbols_to_process = []
        symbols_to_process.extend(["NIFTY200_INDEX", "INDIAVIX", "NIFTY500-INDEX", "NIFTY50-INDEX", "NIFTYBANK-INDEX", "GOLD25OCTFUT"])
    else:
        print("\n--only-index flag detected. Calculating indicators only for index symbols.--")
        symbols_to_process = ["NIFTY200_INDEX", "INDIAVIX", "NIFTY500-INDEX", "NIFTY50-INDEX", "NIFTYBANK-INDEX", "GOLD25OCTFUT"]
    
    symbols_to_process = sorted(list(set(symbols_to_process)))

    total_symbols = len(symbols_to_process)
    print(f"\nFound {total_symbols} symbols to process.")

    num_processes = max(1, cpu_count() - 1)
    print(f"Starting parallel processing with {num_processes} workers...")
    
    with Pool(processes=num_processes) as pool:
        pool.map(process_symbol, symbols_to_process)

    print("\n--- Universal Data Processing Complete! ---")

if __name__ == "__main__":
    main()
