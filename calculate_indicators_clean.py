# calculate_indicators_clean.py
#
# Description:
# This is a corrected and robust version of the original indicator calculation script.
# It reads raw daily data and generates clean, processed data for all required
# timeframes (daily, 2day, weekly, monthly) with all necessary indicators.
#
# MODIFICATION (v6.1 - Added EMA_10 for Monthly Filter):
# 1. ADDED: Calculation for a 10-period EMA to support the 'use_ema_filter'
#    in the monthly simulator.
#
# MODIFICATION (v6.0 - Regime Filter Support):
# 1. ADDED: Calculation for a 200-period EMA to support the new regime filter
#    in the advanced simulators.

import pandas as pd
import os
import sys
import pandas_ta as ta

def calculate_all_indicators(df, rs_period=30, ema_period=30, monthly_ema_period=10, volume_ma_period=20, atr_period=14, atr_ma_period=30, regime_ma_period=50, long_term_ma_period=200):
    """Calculates all required technical indicators on a given dataframe."""
    if df.empty:
        return df
    
    # EMA for strategy trend
    df[f'ema_{ema_period}'] = ta.ema(df['close'], length=ema_period)
    
    # --- NEW: EMA for Monthly Strategy Filter ---
    if monthly_ema_period > 0:
        df[f'ema_{monthly_ema_period}'] = ta.ema(df['close'], length=monthly_ema_period)

    # EMA for market regime filter
    df[f'ema_{regime_ma_period}'] = ta.ema(df['close'], length=regime_ma_period)

    # EMA for Long-Term Regime Filter
    if long_term_ma_period > 0:
        df[f'ema_{long_term_ma_period}'] = ta.ema(df['close'], length=long_term_ma_period)
    
    # Volume MA
    df[f'volume_{volume_ma_period}_sma'] = ta.sma(df['volume'], length=volume_ma_period)
    
    # Relative Strength (simple return)
    df[f'return_{rs_period}'] = df['close'].pct_change(periods=rs_period) * 100
    
    # ATR
    atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    if atr_series is not None and not atr_series.empty:
        df[f'atr_{atr_period}'] = atr_series
        if f'atr_{atr_period}' in df.columns:
            df[f'atr_{atr_ma_period}_ma'] = ta.sma(df[f'atr_{atr_period}'], length=atr_ma_period)
    
    # ATR_6 for monthly simulator
    atr_6_series = ta.atr(df['high'], df['low'], df['close'], length=6)
    if atr_6_series is not None and not atr_6_series.empty:
        df['atr_6'] = atr_6_series
    
    return df

def main():
    """Main function to run the entire data processing pipeline."""
    print("--- Starting Clean Data Processing Engine ---")

    # --- Configuration ---
    input_dir = "historical_data"
    output_base_dir = "data/processed"
    nifty_list_csv = "nifty200.csv"
    
    timeframes = {
        'daily': 'D',
        '2day': '2D',
        'weekly': 'W-FRI',
        'monthly': 'MS'
    }

    # 1. Read the list of stocks
    try:
        stock_list_df = pd.read_csv(nifty_list_csv)
        symbols = stock_list_df["Symbol"].tolist()
        symbols.extend(["NIFTY200_INDEX", "INDIAVIX"]) 
    except FileNotFoundError:
        print(f"Error: '{nifty_list_csv}' not found. Please make sure the file is in the same directory.")
        sys.exit()

    total_stocks = len(symbols)
    print(f"\nFound {total_stocks} symbols to process.")

    # 2. Loop through each symbol
    for i, symbol_name in enumerate(symbols):
        print(f"\nProcessing {symbol_name} ({i+1}/{total_stocks})...")
        
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

        # 5. Use the raw daily data to create and process other higher timeframes
        aggregation_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
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

    print("\n--- Clean Data Processing Complete! ---")
    print(f"All processed files are located in: {output_base_dir}")

if __name__ == "__main__":
    main()
