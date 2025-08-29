# prepare_regime_data.py
#
# Description:
# This script prepares all necessary daily data for implementing advanced
# market regime filters. It is designed to be used by BOTH the long and
# short portfolio simulators.
#
# ENHANCEMENT:
# - Now calculates both bullish and bearish conditions for trend and breadth
#   to make the logic in the simulators cleaner and more explicit.

import os
import pandas as pd
from pytz import timezone

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# --- File Paths ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIRECTORY_DAILY = os.path.join(ROOT_DIR, "data", "universal_processed", "daily")
SYMBOL_LIST_PATH = os.path.join(ROOT_DIR, "nifty200_fno.csv")
OUTPUT_DIRECTORY = os.path.join(ROOT_DIR, "data", "strategy_specific_data")
OUTPUT_FILENAME = "market_regime_data.parquet"

# --- Filter Parameters ---
BREADTH_SMA_PERIOD = 50
NIFTY_TREND_SMA_PERIODS = [20, 30, 50, 100, 200]

# --- Key Symbols ---
NIFTY_50_SYMBOL = 'NIFTY50-INDEX'
INDIA_VIX_SYMBOL = 'INDIAVIX'

def prepare_market_breadth(symbols_to_process, tz):
    """
    Calculates the percentage of stocks above and below their 50-day SMA.
    """
    print("1. Preparing Market Breadth data...")
    all_daily_dfs = []
    for symbol in symbols_to_process:
        daily_path = os.path.join(DATA_DIRECTORY_DAILY, f"{symbol}_daily_with_indicators.parquet")
        if os.path.exists(daily_path):
            df_daily = pd.read_parquet(daily_path)
            if not df_daily.empty:
                df_daily['symbol'] = symbol
                all_daily_dfs.append(df_daily[['symbol', 'close', f'sma_{BREADTH_SMA_PERIOD}']])

    if not all_daily_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_daily_dfs)
    combined_df.dropna(inplace=True)
    
    combined_df['is_below_sma'] = combined_df['close'] < combined_df[f'sma_{BREADTH_SMA_PERIOD}']
    market_breadth = combined_df.groupby(combined_df.index)['is_below_sma'].mean() * 100
    
    market_breadth_df = market_breadth.to_frame(name='breadth_pct_below_sma')
    # NEW: Add the inverse for long strategies
    market_breadth_df['breadth_pct_above_sma'] = 100 - market_breadth_df['breadth_pct_below_sma']
    
    market_breadth_df.index = market_breadth_df.index.tz_localize(tz).normalize()
    print("  - Market Breadth calculation complete.")
    return market_breadth_df


def prepare_volatility_filter(tz):
    """
    Prepares the daily closing price of the India VIX index.
    """
    print("2. Preparing Volatility (India VIX) data...")
    vix_path = os.path.join(DATA_DIRECTORY_DAILY, f"{INDIA_VIX_SYMBOL}_daily_with_indicators.parquet")
    if not os.path.exists(vix_path):
        return pd.DataFrame()

    df_vix = pd.read_parquet(vix_path)
    df_vix = df_vix[['close']].rename(columns={'close': 'india_vix_close'})
    df_vix.index = df_vix.index.tz_localize(tz).normalize()
    print("  - Volatility data preparation complete.")
    return df_vix


def prepare_nifty_trend_filter(tz):
    """
    Calculates whether the NIFTY 50 is trading above or below multiple SMA periods.
    """
    print("3. Preparing NIFTY 50 Trend Filter data...")
    nifty_path = os.path.join(DATA_DIRECTORY_DAILY, f"{NIFTY_50_SYMBOL}_daily_with_indicators.parquet")
    if not os.path.exists(nifty_path):
        return pd.DataFrame()

    df_nifty = pd.read_parquet(nifty_path)
    
    for period in NIFTY_TREND_SMA_PERIODS:
        sma_col = f'sma_{period}'
        if sma_col in df_nifty.columns:
            # Create both below and above columns for clarity
            df_nifty[f'is_nifty_below_sma_{period}'] = df_nifty['close'] < df_nifty[sma_col]
            df_nifty[f'is_nifty_above_sma_{period}'] = df_nifty['close'] > df_nifty[sma_col]
        else:
            df_nifty[f'is_nifty_below_sma_{period}'] = False
            df_nifty[f'is_nifty_above_sma_{period}'] = False

    trend_cols = [col for p in NIFTY_TREND_SMA_PERIODS for col in [f'is_nifty_below_sma_{p}', f'is_nifty_above_sma_{p}']]
    df_nifty_trends = df_nifty[trend_cols]
    df_nifty_trends.index = df_nifty_trends.index.tz_localize(tz).normalize()

    print("  - NIFTY 50 Trend data preparation complete.")
    return df_nifty_trends


def main():
    """Main function to run the data preparation pipeline."""
    print("--- Starting Market Regime Data Preparation ---")
    tz = timezone('Asia/Kolkata')

    try:
        symbols_df = pd.read_csv(SYMBOL_LIST_PATH)
        symbols_to_process = symbols_df['symbol'].dropna().tolist()
    except FileNotFoundError:
        print(f"ERROR: Symbol list not found at {SYMBOL_LIST_PATH}. Cannot calculate market breadth.")
        symbols_to_process = []

    breadth_df = prepare_market_breadth(symbols_to_process, tz)
    volatility_df = prepare_volatility_filter(tz)
    trend_df = prepare_nifty_trend_filter(tz)

    print("\nMerging all regime data...")
    final_df = pd.concat([breadth_df, volatility_df, trend_df], axis=1)
    
    final_df.ffill(inplace=True)
    final_df = final_df[final_df.index.notna()]
    
    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)
    final_df.to_parquet(output_path)

    print(f"\n--- Process Complete! ---")
    print(f"Market regime data saved to: {output_path}")

if __name__ == "__main__":
    main()
