# mae_analyzer.py
#
# Description:
# This script performs a "what-if" analysis on completed backtest trade logs
# to help determine an optimal, data-driven stop-loss level. It analyzes the
# Maximum Adverse Excursion (MAE) of trades to simulate how different, tighter
# stop-losses would have impacted the strategy's performance.
#
# MODIFICATION (v2.2 - Final Recommendation Logic):
# 1. ENHANCED: The recommendation logic has been upgraded to a more robust
#    heuristic. It now finds the tightest stop-loss that preserves at least 95%
#    of original winning trades, providing a more professional "sweet spot" analysis.
#
# MODIFICATION (v2.0 - Multi-Strategy Support):
# 1. ADDED: A '--strategy' command-line argument to specify which simulator
#    to analyze ('htf' for weekly, 'monthly' for monthly).

import pandas as pd
import numpy as np
import os
import argparse
import re

# --- CONFIGURATION ---
STOP_LOSS_TEST_RANGE = np.arange(0.03, 0.16, 0.01)
WINNERS_PRESERVED_THRESHOLD = 95.0 # The minimum percentage of winners to preserve for a SL to be in the "Safe Zone"

def get_log_directory(strategy_name):
    """Returns the correct log directory based on the strategy name."""
    if strategy_name == 'htf':
        return os.path.join('backtest_logs', 'simulator_htf_advanced')
    # Default to monthly
    return os.path.join('backtest_logs', 'simulator_monthly_advanced')

def find_corresponding_summary_file(trade_file_path):
    """
    Finds the matching _summary.txt file for a given _trade_details.csv file
    based on the timestamp in the filename.
    """
    directory = os.path.dirname(trade_file_path)
    base_name = os.path.basename(trade_file_path)
    
    match = re.search(r'(\d{8}_\d{6})', base_name)
    if not match:
        return None, None

    timestamp = match.group(1)
    summary_filename = f"{timestamp}_summary.txt"
    summary_path = os.path.join(directory, summary_filename)

    if os.path.exists(summary_path):
        return summary_path, timestamp
    return None, None

def parse_summary_file(summary_path):
    """
    Parses a _summary.txt file to extract the original performance metrics.
    Handles different formats from different simulators.
    """
    try:
        with open(summary_path, 'r') as f:
            content = f.read()

        # Try parsing the monthly simulator format first
        match = re.search(r"Final Equity: ([\d,.-]+), CAGR: ([\d,.-]+)%, Max Drawdown: ([\d,.-]+)%, Profit Factor: ([\d,.-]+), Win Rate: ([\d,.-]+)%, Total Trades: ([\d,.-]+)", content)
        if match:
            return {
                "CAGR": f"{float(match.group(2)):.2f}%",
                "Max Drawdown": f"{float(match.group(3)):.2f}%",
                "Profit Factor": f"{float(match.group(4)):.2f}",
                "Win Rate": f"{float(match.group(5)):.2f}%",
                "Total Trades": int(match.group(6))
            }
        
        # Fallback to parsing the HTF (weekly) simulator format
        metrics = {}
        patterns = {
            "CAGR": r"CAGR:\s+([\d,.-]+)%",
            "Max Drawdown": r"Max Drawdown:\s+([\d,.-]+)%",
            "Profit Factor": r"Profit Factor:\s+([\d,.-]+)",
            "Win Rate": r"Win Rate:\s+([\d,.-]+)%",
            "Total Trades": r"Total Trades:\s+([\d,.-]+)"
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = match.group(1)
        
        return metrics if metrics else {}

    except Exception:
        return {}

def analyze_mae(trades_df):
    """
    Performs the core MAE "what-if" simulation on a dataframe of trades.
    """
    results = []
    if trades_df.empty or 'mae_percent' not in trades_df.columns:
        return pd.DataFrame()

    trades_df['mae_percent'] = pd.to_numeric(trades_df['mae_percent'], errors='coerce')
    trades_df.dropna(subset=['mae_percent', 'pnl', 'entry_price', 'initial_shares'], inplace=True)
    
    initial_pnl = trades_df['pnl'].sum()

    for sl_percent in STOP_LOSS_TEST_RANGE:
        df_sim = trades_df.copy()
        
        df_sim['hypothetical_loss'] = -sl_percent * df_sim['entry_price'] * df_sim['initial_shares']
        
        stopped_out_mask = df_sim['mae_percent'] > (sl_percent * 100)
        
        df_sim['sim_pnl'] = df_sim['pnl']
        df_sim.loc[stopped_out_mask, 'sim_pnl'] = df_sim['hypothetical_loss']
        
        new_total_pnl = df_sim['sim_pnl'].sum()
        
        winning_trades_new = df_sim[df_sim['sim_pnl'] > 0]
        losing_trades_new = df_sim[df_sim['sim_pnl'] <= 0]
        
        new_win_rate = (len(winning_trades_new) / len(df_sim)) * 100 if not df_sim.empty else 0
        
        gross_profit_new = winning_trades_new['sim_pnl'].sum()
        gross_loss_new = abs(losing_trades_new['sim_pnl'].sum())
        
        new_profit_factor = gross_profit_new / gross_loss_new if gross_loss_new > 0 else np.inf
        
        original_winners = df_sim[df_sim['pnl'] > 0]
        winners_stopped_out = original_winners[original_winners['mae_percent'] > (sl_percent * 100)]
        
        percent_winners_preserved = 100
        if not original_winners.empty:
            percent_winners_preserved = (1 - (len(winners_stopped_out) / len(original_winners))) * 100

        results.append({
            "Stop-Loss Level": f"{sl_percent:.0%}",
            "New Total PnL": f"{new_total_pnl:,.0f}",
            "PnL Preserved": f"{(new_total_pnl / initial_pnl):.1%}" if initial_pnl != 0 else "N/A",
            "New Win Rate": f"{new_win_rate:.1f}%",
            "New Profit Factor": f"{new_profit_factor:.2f}",
            "Original Winners Preserved": f"{percent_winners_preserved:.1f}%"
        })
        
    return pd.DataFrame(results)

def generate_recommendation(results_df):
    """
    Analyzes a results dataframe to provide a smarter, data-driven recommendation
    based on the "Safe Zone" heuristic.
    """
    if results_df.empty:
        return "No data to analyze for a recommendation."

    try:
        # Convert columns to numeric types for analysis
        results_df['winners_preserved_num'] = results_df['Original Winners Preserved'].str.rstrip('%').astype(float)
        results_df['sl_num'] = results_df['Stop-Loss Level'].str.rstrip('%').astype(float)
        
        results_df.dropna(subset=['winners_preserved_num', 'sl_num'], inplace=True)

        # --- NEW HEURISTIC: "Safe Zone" Analysis ---
        # 1. Find all stop levels that preserve at least 95% of original winners.
        safe_zone_df = results_df[results_df['winners_preserved_num'] >= WINNERS_PRESERVED_THRESHOLD]

        if not safe_zone_df.empty:
            # 2. From that safe zone, find the tightest (most efficient) stop-loss.
            optimal_row = safe_zone_df.loc[safe_zone_df['sl_num'].idxmin()]
            
            recommendation = (
                f"\nRECOMMENDATION (Based on 'Safe Zone' Heuristic):\n"
                f"----------------------------------------------------\n"
                f"The most efficient stop-loss in the 'Safe Zone' is around {optimal_row['Stop-Loss Level']}.\n"
                f"This is the tightest stop that still preserves over {WINNERS_PRESERVED_THRESHOLD:.0f}% of the original winning trades ({optimal_row['Original Winners Preserved']}).\n"
                f"It represents a strong balance between risk reduction and preserving profit potential."
            )
            return recommendation
        else:
            return (
                f"\nRECOMMENDATION:\n"
                f"----------------\n"
                f"Could not find an optimal sweet spot. All tested stop-loss levels "
                f"stopped out more than {(100 - WINNERS_PRESERVED_THRESHOLD):.0f}% of the original winning trades."
            )

    except Exception as e:
        return f"\nCould not generate a recommendation due to an error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Analyze Maximum Adverse Excursion (MAE) from backtest trade logs.")
    parser.add_argument('--strategy', type=str, choices=['htf', 'monthly'], default='monthly', help="The strategy to analyze: 'htf' for weekly or 'monthly'. Default is 'monthly'.")
    parser.add_argument('--file', type=str, help="Path to a specific _trade_details.csv file to analyze.")
    args = parser.parse_args()

    log_directory = get_log_directory(args.strategy)

    trade_files = []
    if args.file:
        if os.path.exists(args.file):
            trade_files.append(args.file)
        else:
            print(f"Error: File not found at '{args.file}'")
            return
    else:
        print(f"No specific file provided. Scanning directory: '{log_directory}' for strategy '{args.strategy}'")
        if not os.path.isdir(log_directory):
            print(f"Error: Directory not found.")
            return
        for file in os.listdir(log_directory):
            if file.endswith("_trade_details.csv"):
                trade_files.append(os.path.join(log_directory, file))
    
    if not trade_files:
        print("No trade log files found to analyze.")
        return

    all_trades_df = pd.DataFrame()

    print("\n--- PER-FILE IMPACT ANALYSIS ---")
    for file_path in trade_files:
        print(f"\nAnalyzing: {os.path.basename(file_path)}")
        
        summary_path, timestamp = find_corresponding_summary_file(file_path)
        original_metrics = parse_summary_file(summary_path) if summary_path else {}

        if original_metrics:
            print("Original Performance:")
            for key, value in original_metrics.items():
                print(f"  - {key}: {value}")
        else:
            print("Could not find or parse corresponding summary file.")

        try:
            trades_df = pd.read_csv(file_path)
            all_trades_df = pd.concat([all_trades_df, trades_df], ignore_index=True)
            
            analysis_results = analyze_mae(trades_df)
            if not analysis_results.empty:
                print("\n'What-If' Analysis (Impact of Tighter Stops):")
                print(analysis_results.to_string(index=False))
                
                per_file_recommendation = generate_recommendation(analysis_results)
                print(per_file_recommendation)
            else:
                print("No valid trade data with MAE found in this file.")
        except Exception as e:
            print(f"Could not process file {file_path}. Error: {e}")

    if len(trade_files) > 1:
        print("\n\n--- HOLISTIC IMPACT ANALYSIS (ALL FILES COMBINED) ---")
        if not all_trades_df.empty:
            holistic_results = analyze_mae(all_trades_df)
            if not holistic_results.empty:
                print(holistic_results.to_string(index=False))
                
                holistic_recommendation = generate_recommendation(holistic_results)
                print(holistic_recommendation)
            else:
                print("No valid trade data with MAE found across all files.")
        else:
            print("No trade data was loaded to perform a holistic analysis.")


if __name__ == "__main__":
    main()
