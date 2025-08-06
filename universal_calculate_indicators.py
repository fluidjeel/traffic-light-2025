# universal_calculate_indicators.py
#
# Description:
# A universal indicator calculation script designed to work with the output of
# the 'universal_fyers_scraper.py'.
#
# MODIFICATION (v1.4 - Added Turnover Calculation):
# 1. ADDED: Calculation for daily 'turnover' (close * volume).
# 2. ADDED: Calculation for the 20-day SMA of turnover ('turnover_20_sma').
#
# MODIFICATION (v1.3 - Bug Fix):
# 1. FIXED: A TypeError that occurred during MACD calculation when the input
#    dataframe was too short.

import pandas as pd
import os
import sys
import pandas_ta as ta
import numpy as np

def calculate_all_indicators(df):
    """Calculates all required technical indicators on a given dataframe."""
    if df.empty:
        return df
    
    # --- Trend Indicators ---
    emas = [8, 10, 20, 30, 50, 100, 200]
    for length in emas:
        df[f'ema_{length}'] = ta.ema(df['close'], length=length)

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

    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df['bb_upper_20_2'] = bbands['BBU_20_2.0']
        df['bb_middle_20_2'] = bbands['BBM_20_2.0']
        df['bb_lower_20_2'] = bbands['BBL_20_2.0']

    # --- Volume Indicators ---
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['volume_20_sma'] = ta.sma(df['volume'], length=20)
    df['volume_50_sma'] = ta.sma(df['volume'], length=50)
    
    ### MODIFICATION START: Added Turnover Calculation ###
    df['turnover'] = df['close'] * df['volume']
    df['turnover_20_sma'] = ta.sma(df['turnover'], length=20)
    ### MODIFICATION END ###

    # --- Custom Price Action Indicators ---
    df['52_week_high'] = df['high'].rolling(window=252, min_periods=1).max()
    df['prox_52w_high'] = ((df['close'] - df['52_week_high']) / df['52_week_high']) * 100

    candle_range = df['high'] - df['low']
    body_size = abs(df['close'] - df['open'])
    df['body_ratio'] = np.where(candle_range > 0, body_size / candle_range, 0)
    
    return df

def main():
    """Main function to run the entire data processing pipeline."""
    print("--- Starting Universal Data Processing Engine (with Comprehensive Indicators) ---")

    input_dir = os.path.join("data", "universal_historical_data")
    output_base_dir = "data/universal_processed"
    nifty_list_csv = "nifty200.csv"
    
    timeframes = {'daily': 'D', 'weekly': 'W-FRI', 'monthly': 'MS'}

    try:
        stock_list_df = pd.read_csv(nifty_list_csv)
        symbols = stock_list_df["Symbol"].tolist()
        symbols.extend(["NIFTY200_INDEX", "INDIAVIX"]) 
    except FileNotFoundError:
        print(f"Error: '{nifty_list_csv}' not found."); sys.exit()

    total_symbols = len(symbols)
    print(f"\nFound {total_symbols} symbols to process.")

    for i, symbol_name in enumerate(symbols):
        print(f"\nProcessing {symbol_name} ({i+1}/{total_symbols})...")
        
        input_filename = f"{symbol_name}_daily.csv"
        input_path = os.path.join(input_dir, input_filename)

        if not os.path.exists(input_path):
            print(f"  > Warning: Daily data file not found for {symbol_name}. Skipping.")
            continue

        try:
            df_daily = pd.read_csv(input_path, index_col='datetime', parse_dates=True)
            df_daily = df_daily[~df_daily.index.duplicated(keep='last')]
            df_daily.sort_index(inplace=True)
        except Exception as e:
            print(f"  > Error reading daily data for {symbol_name}: {e}"); continue

        print("  > Calculating indicators for daily timeframe...")
        df_daily_processed = calculate_all_indicators(df_daily.copy())
        
        output_daily_dir = os.path.join(output_base_dir, 'daily')
        os.makedirs(output_daily_dir, exist_ok=True)
        output_daily_path = os.path.join(output_daily_dir, f"{symbol_name}_daily_with_indicators.csv")
        df_daily_processed.to_csv(output_daily_path, index_label='datetime')
        print(f"  > Saved processed daily data to {output_daily_dir}")

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
            print(f"  > Saved processed {tf_name} data to {output_htf_dir}")

    print("\n--- Universal Data Processing Complete! ---")

if __name__ == "__main__":
    main()
