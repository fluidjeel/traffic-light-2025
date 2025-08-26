# mae_mfe_analyzer.py
#
# Description:
# This script automatically finds the latest backtest log and performs an advanced
# analysis on MAE/MFE and the daily RSI at the time of entry. It helps identify
# the most profitable RSI zones for trade signals.
#
# INSTRUCTIONS:
# 1. Place this script in your 'tfl' directory.
# 2. To analyze the LATEST results:
#    python mae_mfe_analyzer.py
# 3. To analyze a SPECIFIC older run:
#    python mae_mfe_analyzer.py --folder 20250824_173000

import pandas as pd
import os
import numpy as np
import argparse
from pytz import timezone

# --- CONFIGURATION ---
# The name of the strategy to analyze. This must match the folder name in 'backtest_logs'.
STRATEGY_NAME = "TrafficLight-Manny-LONGS_ONLY" #"TrafficLight-Manny-SHORTS_ONLY"
# The base directory where all backtest logs are saved.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")
# Path to the 15-minute PARQUET data with pre-calculated indicators.
DATA_DIRECTORY = os.path.join(ROOT_DIR, "data", "universal_processed", "15min")


def find_log_file(specific_folder=None):
    """
    Finds the trade_log.csv file. If a specific folder is provided, it looks
    there. Otherwise, it finds the most recent one.
    """
    strategy_dir = os.path.join(LOGS_BASE_DIR, STRATEGY_NAME)
    if not os.path.isdir(strategy_dir):
        print(f"Error: Strategy directory not found at '{strategy_dir}'")
        return None

    target_run_dir = None
    if specific_folder:
        # Use the user-provided folder name
        target_run_dir = os.path.join(strategy_dir, specific_folder)
        if not os.path.isdir(target_run_dir):
            print(f"Error: The specified folder was not found: {target_run_dir}")
            return None
    else:
        # Find the latest folder automatically
        try:
            all_runs = [d for d in os.listdir(strategy_dir) if os.path.isdir(os.path.join(strategy_dir, d))]
            if not all_runs:
                print(f"Error: No backtest runs found in '{strategy_dir}'")
                return None
            latest_run_folder = sorted(all_runs)[-1]
            target_run_dir = os.path.join(strategy_dir, latest_run_folder)
        except Exception as e:
            print(f"An error occurred while searching for the latest log folder: {e}")
            return None

    log_file_path = os.path.join(target_run_dir, 'trade_log.csv')

    if os.path.exists(log_file_path):
        print(f"Found log file: {log_file_path}\n")
        return log_file_path
    else:
        print(f"Error: 'trade_log.csv' not found in the target directory: {target_run_dir}")
        return None

def analyze_rsi_performance(df):
    """Analyzes the performance of trades based on the daily RSI at entry."""
    if 'daily_rsi_at_entry' not in df.columns:
        print("\nWarning: 'daily_rsi_at_entry' column not found. Skipping RSI analysis.")
        return

    # Create RSI bins
    bins = [0, 30, 40, 50, 60, 70, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '60-70', '>70']
    df['rsi_bucket'] = pd.cut(df['daily_rsi_at_entry'], bins=bins, labels=labels, right=False)

    rsi_analysis = []
    for bucket, group in df.groupby('rsi_bucket'):
        total_trades = len(group)
        if total_trades == 0:
            continue
        
        wins = group[group['pnl'] > 0]
        win_rate = (len(wins) / total_trades) * 100
        
        gross_profit = wins['pnl'].sum()
        gross_loss = abs(group[group['pnl'] <= 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        rsi_analysis.append({
            'RSI Bucket': bucket,
            'Total Trades': total_trades,
            'Win Rate (%)': f"{win_rate:.2f}",
            'Profit Factor': f"{profit_factor:.2f}"
        })
    
    if rsi_analysis:
        print("\n--- RSI Performance Analysis (All Symbols) ---")
        analysis_df = pd.DataFrame(rsi_analysis)
        print(analysis_df.to_string(index=False))
        
        # Suggest the optimal zone
        analysis_df['Profit Factor'] = pd.to_numeric(analysis_df['Profit Factor'])
        best_pf_row = analysis_df.sort_values(by='Profit Factor', ascending=False).iloc[0]
        print(f"\nOptimal RSI Zone Suggestion: The most profitable trades occur in the {best_pf_row['RSI Bucket']} RSI range (Profit Factor: {best_pf_row['Profit Factor']:.2f}).")


def analyze_trades_in_detail(trade_log_df):
    """
    Performs advanced MAE/MFE analysis by cross-referencing price data.
    """
    if 'mfe' not in trade_log_df.columns or 'mae' not in trade_log_df.columns:
        print("Error: 'mfe' and 'mae' columns are required for analysis but not found in the log.")
        return None

    # --- Pre-calculation on the entire log ---
    trade_log_df['initial_risk'] = abs(trade_log_df['entry_price'] - trade_log_df['sl'])
    trade_log_df.loc[trade_log_df['initial_risk'] == 0, 'initial_risk'] = np.nan
    trade_log_df['mfe_r'] = trade_log_df['mfe'] / trade_log_df['initial_risk']
    trade_log_df['mae_r'] = trade_log_df['mae'] / trade_log_df['initial_risk']
    
    trade_log_df['potential_1.5R_plus'] = trade_log_df['mfe_r'] >= 1.5
    trade_log_df['reversal_to_stop'] = (trade_log_df['pnl'] <= 0) & (trade_log_df['mfe'] > 0)

    analysis_results = []
    
    # --- Per-symbol deep dive ---
    for symbol, group in trade_log_df.groupby('symbol'):
        print(f"Analyzing {symbol}...")
        
        data_path = os.path.join(DATA_DIRECTORY, f"{symbol}_15min_with_indicators.parquet")
        if not os.path.exists(data_path):
            print(f"  - Warning: Price data not found for {symbol}. Skipping detailed analysis.")
            continue
        price_df = pd.read_parquet(data_path)
        
        candles_to_0_5R_list = []

        for _, trade in group.iterrows():
            if trade['pnl'] <= 0: continue

            trade_candles = price_df.loc[trade['entry_time']:trade['exit_time']]
            if trade_candles.empty: continue

            target_0_5R = trade['entry_price'] + (trade['initial_risk'] * 0.5) if trade['direction'] == 'LONG' else trade['entry_price'] - (trade['initial_risk'] * 0.5)
            
            if trade['direction'] == 'LONG':
                hit_candles = trade_candles[trade_candles['high'] >= target_0_5R]
            else: # SHORT
                hit_candles = trade_candles[trade_candles['low'] <= target_0_5R]

            if not hit_candles.empty:
                first_hit_time = hit_candles.index[0]
                candles_to_hit = len(trade_candles.loc[trade['entry_time']:first_hit_time])
                candles_to_0_5R_list.append(candles_to_hit)

        symbol_winners = group[group['pnl'] > 0]
        symbol_losers = group[group['pnl'] <= 0]
        
        result = {
            'symbol': symbol,
            'win_count': len(symbol_winners),
            'loss_count': len(symbol_losers),
            'avg_mfe_r': symbol_winners['mfe_r'].mean(),
            'median_mfe_r': symbol_winners['mfe_r'].median(),
            'potential_1.5R_plus_wins': symbol_winners['potential_1.5R_plus'].sum(),
            'reversal_to_stop_count': symbol_losers['reversal_to_stop'].sum(),
            'avg_candles_to_0.5R': np.mean(candles_to_0_5R_list) if candles_to_0_5R_list else 0
        }
        analysis_results.append(result)

    return pd.DataFrame(analysis_results)


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Advanced MAE/MFE & RSI Analyzer for backtest logs.")
    parser.add_argument('--folder', type=str, required=False, help='Specify the timestamped folder name of a specific backtest run to analyze.')
    args = parser.parse_args()

    log_file = find_log_file(specific_folder=args.folder)
    if not log_file:
        return

    try:
        trade_df = pd.read_csv(log_file, parse_dates=['entry_time', 'exit_time'])
        
        # --- Timezone Standardization ---
        if pd.api.types.is_datetime64_any_dtype(trade_df['entry_time']) and trade_df['entry_time'].dt.tz is not None:
            trade_df['entry_time'] = trade_df['entry_time'].dt.tz_localize(None)
        if pd.api.types.is_datetime64_any_dtype(trade_df['exit_time']) and trade_df['exit_time'].dt.tz is not None:
            trade_df['exit_time'] = trade_df['exit_time'].dt.tz_localize(None)

    except Exception as e:
        print(f"Error reading trade log file: {e}")
        return

    if trade_df.empty:
        print("Trade log is empty. No analysis to perform.")
        return

    analysis_df = analyze_trades_in_detail(trade_df)

    if analysis_df is None or analysis_df.empty:
        print("Analysis could not be completed.")
    else:
        print("\n--- Advanced MAE / MFE Analysis Results ---")
        for _, row in analysis_df.iterrows():
            print(f"\n-------------------- {row['symbol']} --------------------")
            
            total_trades = row['win_count'] + row['loss_count']
            print(f"Total Trades: {total_trades} (Wins: {row['win_count']}, Losses: {row['loss_count']})")
            
            if row['win_count'] > 0:
                potential_pct = (row['potential_1.5R_plus_wins'] / row['win_count']) * 100
                print("\n  Winning Trades Insights:")
                print(f"    - Median MFE: {row['median_mfe_r']:.2f}R (50% of winners ran further than this)")
                print(f"    - Potential for >1.5R: {row['potential_1.5R_plus_wins']} of {row['win_count']} wins ({potential_pct:.1f}%) had the potential to be big winners.")
                if row['avg_candles_to_0.5R'] > 0:
                    avg_time_to_profit = row['avg_candles_to_0.5R'] * 15
                    print(f"    - Avg Time to 0.5R Profit: {row['avg_candles_to_0.5R']:.1f} candles (~{avg_time_to_profit:.0f} minutes)")

            if row['loss_count'] > 0:
                reversal_pct = (row['reversal_to_stop_count'] / row['loss_count']) * 100
                print("\n  Losing Trades Insights:")
                print(f"    - Reversals to Stop: {row['reversal_to_stop_count']} of {row['loss_count']} losses ({reversal_pct:.1f}%) were profitable before reversing.")

        analysis_filename = os.path.join(os.path.dirname(log_file), 'advanced_analysis.csv')
        analysis_df.to_csv(analysis_filename, index=False)
        print(f"\n\nDetailed MAE/MFE analysis saved to: {analysis_filename}")

    # --- Run RSI Analysis on the full trade log ---
    analyze_rsi_performance(trade_df)


if __name__ == "__main__":
    main()
