# volume_analyzer.py
#
# Description:
# This script performs a "Winners vs. Losers" analysis to determine if a
# volume-based filter could improve the strategy's robustness.
#
# It answers the question: "Do my winning trades consistently occur on higher
# relative volume than my losing trades?"
#
# How it works:
# 1. Loads the trade_log.csv from the latest backtest run.
# 2. Enriches the trade log with the volume ratio (volume / 20-period SMA)
#    that was present at the time of each trade's entry.
# 3. Separates all trades into two groups: Winners (PnL > 0) and Losers (PnL <= 0).
# 4. Calculates and compares the statistical distribution of the volume ratio
#    for both groups to identify any significant differences.

import os
import pandas as pd
from pytz import timezone
import glob

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# --- File Paths ---
# Assumes this script is in the /tfl folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")
MASTER_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_shorts_data_with_signals.parquet")

INDIA_TZ = timezone('Asia/Kolkata')

def find_latest_run_directory(base_log_dir):
    """Finds the most recent backtest run directory."""
    list_of_dirs = [d for d in glob.glob(os.path.join(base_log_dir, '*/*')) if os.path.isdir(d)]
    if not list_of_dirs:
        return None
    return max(list_of_dirs, key=os.path.getmtime)

def main():
    """Main function to run the analysis."""
    print("--- Starting Volume Winners vs. Losers Analysis ---")

    # --- 1. Find and Load Data ---
    latest_run_dir = find_latest_run_directory(LOGS_BASE_DIR)
    if not latest_run_dir:
        print(f"ERROR: No backtest run directories found in {LOGS_BASE_DIR}")
        return
    
    print(f"Analyzing latest run: {os.path.basename(latest_run_dir)}")
    trade_log_path = os.path.join(latest_run_dir, "trade_log.csv")

    if not os.path.exists(trade_log_path):
        print(f"ERROR: trade_log.csv not found in {latest_run_dir}.")
        return
        
    df_trades = pd.read_csv(trade_log_path, parse_dates=['entry_time'])
    df_trades['entry_time'] = df_trades['entry_time'].dt.tz_convert(INDIA_TZ)

    # --- 2. Enrich Trade Log with Volume Data ---
    print("Loading master data file to get volume ratio at entry...")
    df_master = pd.read_parquet(MASTER_DATA_PATH)
    df_master['datetime'] = pd.to_datetime(df_master['datetime']).dt.tz_convert(INDIA_TZ)
    
    # Create a unique key for merging
    df_trades['merge_key'] = df_trades['symbol'].astype(str) + df_trades['entry_time'].astype(str)
    df_master['merge_key'] = df_master['symbol'].astype(str) + df_master['datetime'].astype(str)

    # Merge the volume_ratio onto the trade log
    df_trades = pd.merge(
        df_trades,
        df_master[['merge_key', 'volume_ratio']],
        on='merge_key',
        how='left'
    )
    df_trades.drop(columns=['merge_key'], inplace=True)
    df_trades.dropna(subset=['volume_ratio'], inplace=True)

    if df_trades.empty:
        print("Could not merge volume data. Aborting analysis.")
        return

    # --- 3. Separate into Winners and Losers ---
    df_winners = df_trades[df_trades['pnl'] > 0]
    df_losers = df_trades[df_trades['pnl'] <= 0]

    if df_winners.empty or df_losers.empty:
        print("Not enough data for a meaningful winners vs. losers analysis.")
        return

    # --- 4. Perform Statistical Analysis ---
    print("\n--- Volume Analysis Results ---")
    
    stats_winners = df_winners['volume_ratio'].describe()
    stats_losers = df_losers['volume_ratio'].describe()

    print("\n** Volume Ratio at Entry for WINNING Trades **")
    print(f"  - Average Ratio: {stats_winners['mean']:.2f}")
    print(f"  - Median Ratio:  {stats_winners['50%']:.2f}")
    print(f"  - Ratio Range (25th-75th percentile): {stats_winners['25%']:.2f} - {stats_winners['75%']:.2f}")

    print("\n** Volume Ratio at Entry for LOSING Trades **")
    print(f"  - Average Ratio: {stats_losers['mean']:.2f}")
    print(f"  - Median Ratio:  {stats_losers['50%']:.2f}")
    print(f"  - Ratio Range (25th-75th percentile): {stats_losers['25%']:.2f} - {stats_losers['75%']:.2f}")

    # --- 5. Conclusion ---
    print("\n--- Conclusion ---")
    if stats_winners['mean'] > stats_losers['mean']:
        print("✅ The analysis suggests a VOLUME FILTER could be effective.")
        print("   Winning trades, on average, occurred on higher relative volume than losing trades.")
        print(f"   Consider adding a filter like: 'Only trade if volume_ratio > {stats_losers['75%']:.2f}'")
    else:
        print("❌ The analysis does NOT show a clear edge for a volume filter.")
        print("   The average volume ratio for winners and losers was very similar.")

if __name__ == "__main__":
    main()
