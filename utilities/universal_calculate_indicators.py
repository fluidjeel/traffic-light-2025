# universal_calculate_indicators.py (Enhanced for Portability and Progress Bar)
#
# Description:
# A universal indicator calculation script designed to work with the output of the scraper.
#
# NEW FEATURES:
# - Portability: Can be run from a subdirectory (e.g., 'utilities') with paths relative to the project root.
# - Progress Bar: Replaced verbose console output with a clean tqdm progress bar.
# - Graceful Exit: Handles Ctrl+C to terminate gracefully.
# - Data Consistency: Aligned CSV reading/writing with the scraper (datetime as a column, not index).

import pandas as pd
import os
import sys
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import signal

# --- ENHANCEMENT: Project Root Configuration for Portability ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

# --- COMPATIBILITY FIX for pandas_ta ---
np.NaN = np.nan
try:
    import pandas_ta as ta
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! FATAL ERROR: pandas_ta library NOT FOUND !!!")
    print("!!! Please install it using: pip install pandas_ta tqdm !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sys.exit(1)

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
    df['return_30'] = df['close'].pct_change(periods=30) * 100

    # --- Volatility Indicators ---
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_14_pct'] = (df['atr_14'] / df['close']) * 100
    df['atr_6'] = ta.atr(df['high'], df['low'], df['close'], length=6)
    df['vix_10_sma'] = ta.sma(df['close'], length=10)

    # --- Volume Indicators ---
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['volume_20_sma'] = ta.sma(df['volume'], length=20)
    df['volume_50_sma'] = ta.sma(df['volume'], length=50)
    df['volume_ratio'] = df['volume'] / df['volume_20_sma']
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
    df['volatility_ratio'] = df['atr_14'] / df['ema_20'] if 'ema_20' in df.columns else np.nan
    
    return df

def process_symbol(symbol_name):
    """Processes all timeframes for a single symbol. Called by each parallel worker."""
    input_dir = os.path.join(PROJECT_ROOT, "data", "universal_historical_data")
    output_base_dir = os.path.join(PROJECT_ROOT, "data", "universal_processed")
    timeframes = {'weekly': 'W-FRI', 'monthly': 'MS'}

    # --- Daily Processing ---
    input_path_daily = os.path.join(input_dir, f"{symbol_name}_daily.csv")
    if not os.path.exists(input_path_daily):
        return
    
    try:
        df_daily = pd.read_csv(input_path_daily, parse_dates=['datetime'])
        if df_daily.empty: return

        df_daily = df_daily.drop_duplicates(subset='datetime').sort_values('datetime')
    except Exception:
        return

    df_daily_processed = calculate_all_indicators(df_daily.copy())
    output_daily_dir = os.path.join(output_base_dir, 'daily')
    os.makedirs(output_daily_dir, exist_ok=True)
    df_daily_processed.to_csv(os.path.join(output_daily_dir, f"{symbol_name}_daily_with_indicators.csv"), index=False)

    # --- Higher Timeframe Processing (from Daily) ---
    aggregation_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    df_daily_indexed = df_daily.set_index('datetime')
    for tf_name, rule in timeframes.items():
        df_htf = df_daily_indexed.resample(rule).agg(aggregation_rules).dropna().reset_index()
        if df_htf.empty: continue
        
        df_htf_processed = calculate_all_indicators(df_htf)
        output_htf_dir = os.path.join(output_base_dir, tf_name)
        os.makedirs(output_htf_dir, exist_ok=True)
        df_htf_processed.to_csv(os.path.join(output_htf_dir, f"{symbol_name}_{tf_name}_with_indicators.csv"), index=False)

    # --- 15-Minute Processing ---
    input_path_15min = os.path.join(input_dir, f"{symbol_name}_15min.csv")
    if os.path.exists(input_path_15min):
        try:
            df_15min = pd.read_csv(input_path_15min, parse_dates=['datetime'])
            if df_15min.empty: return

            df_15min = df_15min.drop_duplicates(subset='datetime').sort_values('datetime')
            df_15min_processed = calculate_all_indicators(df_15min.copy())
            
            output_15min_dir = os.path.join(output_base_dir, '15min')
            os.makedirs(output_15min_dir, exist_ok=True)
            df_15min_processed.to_csv(os.path.join(output_15min_dir, f"{symbol_name}_15min_with_indicators.csv"), index=False)
        except Exception:
            return

def worker_init():
    """Initializer for worker processes to ignore interrupt signals."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    """Main function to run the entire data processing pipeline in parallel."""
    parser = argparse.ArgumentParser(description="Universal Indicator Calculator")
    parser.add_argument('--only-index', action='store_true', help='Calculate indicators only for the indices.')
    args = parser.parse_args()

    nifty_list_csv = os.path.join(PROJECT_ROOT, "nifty500.csv")
    index_list = ["NIFTY200_INDEX", "INDIAVIX", "NIFTY500-INDEX", "NIFTY50-INDEX", "NIFTYBANK-INDEX", "GOLD25OCTFUT"]
    
    symbols_to_process = []
    if not args.only_index:
        try:
            symbols_to_process = pd.read_csv(nifty_list_csv)["Symbol"].dropna().tolist()
        except FileNotFoundError:
            print(f"Warning: '{nifty_list_csv}' not found. Processing indices only.")
    symbols_to_process.extend(index_list)
    symbols_to_process = sorted(list(set(symbols_to_process)))

    if not symbols_to_process:
        print("No symbols to process. Exiting.")
        return

    print(f"Found {len(symbols_to_process)} unique symbols to process.")

    num_processes = max(1, cpu_count() - 1)
    
    try:
        with Pool(processes=num_processes, initializer=worker_init) as pool:
            desc = "Calculating Indicators"
            # Use tqdm with imap_unordered for a real-time progress bar
            for _ in tqdm(pool.imap_unordered(process_symbol, symbols_to_process), total=len(symbols_to_process), desc=desc):
                pass
    except KeyboardInterrupt:
        print("\n\n!!! KeyboardInterrupt detected. Terminating... !!!")
        sys.exit(1)

    print("\n--- Universal Data Processing Complete! ---")

if __name__ == "__main__":
    main()

