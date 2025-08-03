# weekly_mae_mfe_vix_analyzer.py
#
# Description:
# An advanced "what-if" analyzer for the weekly TFL strategy. It ingests trade
# logs and simulates performance across a matrix of different stop-loss levels
# and trailing stop-loss parameters to find the optimal exit strategy.
#
# MODIFICATION (v1.4 - Upside Capture Analysis for Trailing Stop):
# 1. ADDED: Calculation of "Optimized Upside Capture Ratio" to quantify how
#    much potential profit the optimized trailing stop strategy captures.
# 2. UPDATED: The 'run_what_if_simulation' function now also returns the
#    average P&L of only the winning trades in the simulation.
# 3. UPDATED: The console output to show a direct comparison between the
#    baseline and the optimized strategy's profit capture efficiency.
#
# MODIFICATION (v1.3 - Trailing Stop Simulation):
# 1. ADDED: The ability to simulate a trailing stop-loss mechanism.

import pandas as pd
import numpy as np
import os
import glob
import argparse
from datetime import datetime

def calculate_performance_metrics(pnl_percentages, risk_free_rate=0.0):
    """Calculates Sharpe Ratio from a series of P&L percentages."""
    if len(pnl_percentages) < 2:
        return 0.0

    pnl_array = np.array(pnl_percentages) / 100
    excess_returns = pnl_array - (risk_free_rate / 252)
    mean_return = excess_returns.mean()
    std_dev = excess_returns.std()

    if std_dev == 0:
        return 0.0

    sharpe_ratio = mean_return / std_dev
    annualized_sharpe = sharpe_ratio * np.sqrt(252) 
    return annualized_sharpe

def run_what_if_simulation(df, stop_loss_pct, activation_pct, trail_pct):
    """
    Simulates trade outcomes with a hypothetical stop-loss and trailing stop.
    """
    if df.empty:
        return 0, 0, 0, 0, 0, 0, 0

    simulated_pnl = []

    for _, trade in df.iterrows():
        if trade['mae_percent'] >= stop_loss_pct * 100:
            simulated_pnl.append(-stop_loss_pct * 100)
            continue

        if trade['mfe_percent'] >= activation_pct * 100:
            exit_price_pct = trade['mfe_percent'] - (trail_pct * 100)
            final_pnl = max(exit_price_pct, -stop_loss_pct * 100)
            simulated_pnl.append(final_pnl)
        else:
            simulated_pnl.append(trade['captured_pct'])
            
    simulated_pnl = np.array(simulated_pnl)
    
    wins = simulated_pnl > 0
    losses = simulated_pnl < 0
    
    win_rate = np.sum(wins) / len(simulated_pnl) if len(simulated_pnl) > 0 else 0
    gross_profit = np.sum(simulated_pnl[wins])
    gross_loss = abs(np.sum(simulated_pnl[losses]))
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    avg_pnl_pct = simulated_pnl.mean()
    
    # MODIFICATION: Calculate avg pnl for winning trades only
    avg_pnl_winners = simulated_pnl[wins].mean() if np.sum(wins) > 0 else 0
    
    sharpe_ratio = calculate_performance_metrics(simulated_pnl)
    
    return profit_factor, win_rate * 100, avg_pnl_pct, avg_pnl_winners, sharpe_ratio, simulated_pnl.sum(), len(simulated_pnl)

def analyze_trades(strategy_name):
    """
    Main function to load trade logs and run the analysis.
    """
    print(f"--- Starting VIX-Aware MAE/MFE Analysis for '{strategy_name}' ---")
    
    log_folder = os.path.join('backtest_logs', strategy_name)
    trade_files = glob.glob(os.path.join(log_folder, '*_trade_details.csv'))

    if not trade_files:
        print(f"Error: No trade logs found in '{log_folder}'. Please run the simulator first.")
        return

    all_trades_df = pd.concat([pd.read_csv(f) for f in trade_files], ignore_index=True)
    
    required_cols = ['vix_on_entry', 'mae_percent', 'mfe_percent', 'captured_pct', 'pnl']
    if not all(col in all_trades_df.columns for col in required_cols):
        print(f"Error: Trade log is missing required columns. Please use the latest simulator version.")
        print(f"Missing: {[col for col in required_cols if col not in all_trades_df.columns]}")
        return
        
    print(f"Loaded {len(all_trades_df)} total trade events from {len(trade_files)} log file(s).")

    vix_buckets = {
        'Calm (VIX < 15)': (0, 15),
        'Moderate (VIX 15-22)': (15, 22),
        'High (VIX 22-30)': (22, 30),
        'Extreme (VIX > 30)': (30, 999)
    }
    
    stop_loss_range = np.arange(0.05, 0.16, 0.02)
    activation_range = np.arange(0.05, 0.21, 0.05)
    trail_range = np.arange(0.03, 0.13, 0.02)

    all_results = []
    
    for bucket_name, (vix_min, vix_max) in vix_buckets.items():
        print(f"\n--- Analyzing VIX Regime: {bucket_name} ---")
        
        bucket_df = all_trades_df[
            (all_trades_df['vix_on_entry'] >= vix_min) &
            (all_trades_df['vix_on_entry'] < vix_max)
        ].copy()

        if bucket_df.empty:
            print("No trades found in this VIX regime. Skipping.")
            continue
        
        print(f"Found {len(bucket_df)} trades in this regime.")
        
        # --- Baseline Analysis ---
        original_winners = bucket_df[bucket_df['pnl'] > 0]
        avg_mfe_baseline = 0
        if not original_winners.empty:
            avg_mfe_baseline = original_winners['mfe_percent'].mean()
            avg_captured_baseline = original_winners['captured_pct'].mean()
            baseline_capture_ratio = (avg_captured_baseline / avg_mfe_baseline) * 100 if avg_mfe_baseline > 0 else 0
            
            print("\n  Original Trade Performance (Baseline):")
            print(f"  - Avg Potential Upside (MFE) on Winners: {avg_mfe_baseline:.2f}%")
            print(f"  - Upside Capture Ratio: {baseline_capture_ratio:.1f}%")
        else:
            print("\n  Original Trade Performance (Baseline): No winning trades to analyze.")
        print("  -----------------------------------------")

        best_sharpe = -np.inf
        best_params = {}

        for sl in stop_loss_range:
            for act_pct in activation_range:
                for trail_pct in trail_range:
                    if act_pct <= trail_pct:
                        continue

                    pf, wr, avg_pnl, avg_pnl_winners, sharpe, total_pnl, num_trades = run_what_if_simulation(bucket_df, sl, act_pct, trail_pct)
                    
                    result = {
                        'vix_regime': bucket_name,
                        'stop_loss_pct': sl * 100,
                        'activation_pct': act_pct * 100,
                        'trail_pct': trail_pct * 100,
                        'sharpe_ratio': sharpe,
                        'profit_factor': pf,
                        'win_rate_pct': wr,
                        'avg_simulated_pnl_winners': avg_pnl_winners,
                        'trade_count': num_trades
                    }
                    all_results.append(result)

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = result
        
        if best_params:
            print("\n  Optimal Trailing Stop Parameters Found (based on Sharpe Ratio):")
            print(f"  - Initial Stop-Loss: {best_params['stop_loss_pct']:.1f}%")
            print(f"  - Trail Activation at: {best_params['activation_pct']:.1f}% profit")
            print(f"  - Trail Amount: {best_params['trail_pct']:.1f}%")
            print(f"  - Resulting Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
            
            # MODIFICATION: Calculate and display the optimized capture ratio
            if avg_mfe_baseline > 0:
                optimized_capture_ratio = (best_params['avg_simulated_pnl_winners'] / avg_mfe_baseline) * 100
                print(f"  - Optimized Upside Capture Ratio: {optimized_capture_ratio:.1f}%")


    if not all_results:
        print("\nNo results generated.")
        return
        
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(log_folder, f'{timestamp}_vix_aware_trailing_stop_analysis.csv')
    results_df.to_csv(output_filename, index=False, float_format='%.2f')
    
    print(f"\n--- Analysis Complete ---")
    print(f"Detailed simulation grid saved to: {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VIX-Aware Trailing Stop Analyzer for Weekly TFL Strategy")
    parser.add_argument('--strategy', type=str, required=True, help='The name of the strategy folder in backtest_logs.')
    args = parser.parse_args()
    
    analyze_trades(args.strategy)
