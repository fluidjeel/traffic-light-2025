# universal_calculate_indicators.py
#
# Description:
# A universal indicator calculation script designed to work with the output of
# the 'universal_fyers_scraper.py'. It reads the raw daily data from the
# unified historical data folder, calculates all necessary indicators, resamples
# to higher timeframes, and saves the processed files for the simulators.
#
# MODIFICATION (v1.0):
# - Updated input directory and filename conventions to align with the
#   universal data pipeline.
# - Preserved all core calculation logic from the previous version.

import pandas as pd
import os
import sys
import pandas_ta as ta

def calculate_all_indicators(df, rs_period=30, ema_period=30, monthly_ema_period=10, volume_ma_period=20, atr_period=14, atr_ma_period=30, regime_ma_period=50, long_term_ma_period=200):
    """Calculates all required technical indicators on a given dataframe."""
    if df.empty:
        return df
    
    # Standard EMA for daily/weekly strategies
    df[f'ema_{ema_period}'] = ta.ema(df['close'], length=ema_period)
    
    # EMA for the monthly strategy's trend filter
    if monthly_ema_period > 0:
        df[f'ema_{monthly_ema_period}'] = ta.ema(df['close'], length=monthly_ema_period)

    # EMA for the daily market regime filter
    df[f'ema_{regime_ma_period}'] = ta.ema(df['close'], length=regime_ma_period)

    # EMA for the long-term market regime filter
    if long_term_ma_period > 0:
        df[f'ema_{long_term_ma_period}'] = ta.ema(df['close'], length=long_term_ma_period)
    
    # Volume Simple Moving Average
    df[f'volume_{volume_ma_period}_sma'] = ta.sma(df['volume'], length=volume_ma_period)
    
    # Relative Strength (simple percentage return)
    df[f'return_{rs_period}'] = df['close'].pct_change(periods=rs_period) * 100
    
    # Standard ATR
    atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    if atr_series is not None and not atr_series.empty:
        df[f'atr_{atr_period}'] = atr_series
        if f'atr_{atr_period}' in df.columns:
            df[f'atr_{atr_ma_period}_ma'] = ta.sma(df[f'atr_{atr_period}'], length=atr_ma_period)
    
    # ATR with a 6-period length for the monthly simulator
    atr_6_series = ta.atr(df['high'], df['low'], df['close'], length=6)
    if atr_6_series is not None and not atr_6_series.empty:
        df['atr_6'] = atr_6_series
    
    return df

def main():
    """Main function to run the entire data processing pipeline."""
    print("--- Starting Universal Data Processing Engine ---")

    # --- Configuration ---
    # UPDATED: Pointing to the new universal scraper's output directory
    input_dir = os.path.join("data", "universal_historical_data")
    output_base_dir = "data/universal_processed"
    nifty_list_csv = "nifty200.csv"
    
    timeframes = {
        'daily': 'D',
        '2day': '2D',
        'weekly': 'W-FRI',
        'monthly': 'MS'
    }

    # 1. Read the list of stocks and indices
    try:
        stock_list_df = pd.read_csv(nifty_list_csv)
        symbols = stock_list_df["Symbol"].tolist()
        symbols.extend(["NIFTY200_INDEX", "INDIAVIX"]) 
    except FileNotFoundError:
        print(f"Error: '{nifty_list_csv}' not found. Please make sure the file is in the same directory.")
        sys.exit()

    total_symbols = len(symbols)
    print(f"\nFound {total_symbols} symbols to process.")

    # 2. Loop through each symbol
    for i, symbol_name in enumerate(symbols):
        print(f"\nProcessing {symbol_name} ({i+1}/{total_symbols})...")
        
        # UPDATED: Using the new filename convention from the universal scraper
        input_filename = f"{symbol_name}_daily.csv"
        input_path = os.path.join(input_dir, input_filename)

        if not os.path.exists(input_path):
            print(f"  > Warning: Daily data file not found for {symbol_name} at {input_path}. Skipping.")
            continue

        # 3. Read and prepare the daily data
        try:
            df_daily = pd.read_csv(input_path, index_col='datetime', parse_dates=True)
            df_daily = df_daily[~df_daily.index.duplicated(keep='last')]
            df_daily.sort_index(inplace=True)
            if df_daily.empty:
                print(f"  > Warning: Daily data file for {symbol_name} is empty. Skipping.")
                continue
        except Exception as e:
            print(f"  > Error reading daily data for {symbol_name}: {e}")
            continue

        # 4. Calculate indicators on the daily data first
        print("  > Calculating indicators for daily timeframe...")
        df_daily_processed = calculate_all_indicators(df_daily.copy())
        
        output_daily_dir = os.path.join(output_base_dir, 'daily')
        os.makedirs(output_daily_dir, exist_ok=True)
        output_daily_path = os.path.join(output_daily_dir, f"{symbol_name}_daily_with_indicators.csv")
        df_daily_processed.to_csv(output_daily_path, index_label='datetime')
        print(f"  > Saved processed daily data to {output_daily_dir}")

        # 5. Resample to higher timeframes and calculate indicators
        aggregation_rules = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }

        for tf_name, rule in timeframes.items():
            if tf_name == 'daily':
                continue

            print(f"  > Aggregating to {tf_name} timeframe...")
            df_htf = df_daily.resample(rule).agg(aggregation_rules).dropna()
            
            if df_htf.empty:
                print(f"  > Warning: No {tf_name} data after resampling. Skipping.")
                continue
            
            df_htf = df_htf[~df_htf.index.duplicated(keep='last')]
            
            print(f"  > Calculating indicators for {tf_name} timeframe...")
            df_htf_processed = calculate_all_indicators(df_htf)
            
            output_htf_dir = os.path.join(output_base_dir, tf_name)
            os.makedirs(output_htf_dir, exist_ok=True)
            output_htf_path = os.path.join(output_htf_dir, f"{symbol_name}_{tf_name}_with_indicators.csv")
            df_htf_processed.to_csv(output_htf_path, index_label='datetime')
            print(f"  > Saved processed {tf_name} data to {output_htf_dir}")

    print("\n--- Universal Data Processing Complete! ---")
    print(f"All processed files are located in: {output_base_dir}")

if __name__ == "__main__":
    main()
