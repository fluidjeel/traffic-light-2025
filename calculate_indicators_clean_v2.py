# calculate_indicators_clean.py
#
# Description:
# This is a corrected and robust version of the original indicator calculation script.
#
# MODIFICATION (v3.0 - Monthly Strategy Enhancement):
# 1. ADDED: Calculation for a 10-period EMA to support the monthly strategy's
#    primary trend filter.
# 2. ADDED: Calculation for a 12-period SMA of volume to support the monthly
#    setup volume filter (12-month average).
# 3. ADDED: Calculation for a 6-period ATR to support the monthly strategy's
#    new volatility-adjusted stop-loss logic (6-month ATR).

import pandas as pd
import os
import sys
import pandas_ta as ta

def calculate_all_indicators(df, rs_period=30, ema_period=30, volume_ma_period=20, atr_period=14, atr_ma_period=30, regime_ma_period=50):
    """Calculates all required technical indicators on a given dataframe."""
    if df.empty:
        return df
    
    # --- Standard Indicators ---
    df[f'ema_{ema_period}'] = ta.ema(df['close'], length=ema_period)
    df[f'ema_{regime_ma_period}'] = ta.ema(df['close'], length=regime_ma_period)
    df[f'volume_{volume_ma_period}_sma'] = ta.sma(df['volume'], length=volume_ma_period)
    df[f'return_{rs_period}'] = df['close'].pct_change(periods=rs_period) * 100
    
    # --- Monthly Strategy Indicators ---
    df['ema_10'] = ta.ema(df['close'], length=10)
    df['volume_12_sma'] = ta.sma(df['volume'], length=12)
    df['atr_6'] = ta.atr(df['high'], df['low'], df['close'], length=6)

    # --- Standard ATR ---
    atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    if atr_series is not None and not atr_series.empty:
        df[f'atr_{atr_period}'] = atr_series
        if f'atr_{atr_period}' in df.columns:
            df[f'atr_{atr_ma_period}_ma'] = ta.sma(df[f'atr_{atr_period}'], length=atr_ma_period)

    return df

def main():
    """Main function to run the entire data processing pipeline."""
    print("--- Starting Clean Data Processing Engine (v3) ---")

    input_dir = "historical_data"
    output_base_dir = "data/processed"
    nifty_list_csv = "nifty200.csv"
    
    timeframes = {
        'daily': 'D',
        'weekly': 'W-FRI',
        'monthly': 'MS'
    }

    try:
        stock_list_df = pd.read_csv(nifty_list_csv)
        symbols = stock_list_df["Symbol"].tolist()
        symbols.extend(["NIFTY200_INDEX", "INDIAVIX"]) 
    except FileNotFoundError:
        print(f"Error: '{nifty_list_csv}' not found.")
        sys.exit()

    total_stocks = len(symbols)
    print(f"\nFound {total_stocks} symbols to process.")

    for i, symbol_name in enumerate(symbols):
        print(f"\nProcessing {symbol_name} ({i+1}/{total_stocks})...")
        
        input_path = os.path.join(input_dir, f"{symbol_name}_daily.csv")
        if not os.path.exists(input_path):
            print(f"  > Warning: Daily data file not found for {symbol_name}. Skipping.")
            continue

        try:
            df_daily = pd.read_csv(input_path, index_col='datetime', parse_dates=True)
            df_daily.sort_index(inplace=True)
            if df_daily.empty:
                print(f"  > Warning: Daily data file for {symbol_name} is empty. Skipping.")
                continue
        except Exception as e:
            print(f"  > Error reading daily data for {symbol_name}: {e}")
            continue

        print("  > Calculating indicators for daily timeframe...")
        df_daily_processed = calculate_all_indicators(df_daily.copy())
        output_daily_dir = os.path.join(output_base_dir, 'daily')
        os.makedirs(output_daily_dir, exist_ok=True)
        df_daily_processed.to_csv(os.path.join(output_daily_dir, f"{symbol_name}_daily_with_indicators.csv"), index_label='datetime')
        print(f"  > Saved processed daily data to {output_daily_dir}")

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
                
            print(f"  > Calculating indicators for {tf_name} timeframe...")
            df_htf_processed = calculate_all_indicators(df_htf)
            
            output_htf_dir = os.path.join(output_base_dir, tf_name)
            os.makedirs(output_htf_dir, exist_ok=True)
            df_htf_processed.to_csv(os.path.join(output_htf_dir, f"{symbol_name}_{tf_name}_with_indicators.csv"), index_label='datetime')
            print(f"  > Saved processed {tf_name} data to {output_htf_dir}")

    print("\n--- Clean Data Processing Complete! ---")

if __name__ == "__main__":
    main()
