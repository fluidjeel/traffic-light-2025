# daily_mae_mfe_analyzer.py
#
# Description:
# An advanced "what-if" analyzer for the Daily TFL strategy. It ingests trade
# logs that contain VIX-on-entry and full excursion data (MAE/MFE).
# The script segments trades by VIX regimes and simulates performance across a
# matrix of different stop-loss lookback periods and R:R profit target
# multipliers to find the optimal parameters for each market condition.
#
# Usage:
# python daily_mae_mfe_analyzer.py --strategy daily_tfl_simulator

import pandas as pd
import numpy as np
import os
import glob
import argparse
from datetime import datetime

def calculate_sharpe_ratio(pnl_percentages, risk_free_rate=0.0):
    """Calculates a simple annualized Sharpe Ratio from a series of P&L percentages."""
    if len(pnl_percentages) < 2:
        return 0.0

    pnl_array = np.array(pnl_percentages) / 100
    excess_returns = pnl_array - (risk_free_rate / 252)
    mean_return = excess_returns.mean()
    std_dev = excess_returns.std()

    if std_dev == 0: return 0.0
    
    return (mean_return / std_dev) * np.sqrt(252)

def run_what_if_simulation(df, stop_loss_pct, profit_target_pct):
    """
    Simulates trade outcomes for a given stop-loss and profit target percentage.
    """
    if df.empty:
        return 0, 0, 0

    simulated_pnl = []
    for _, trade in df.iterrows():
        if trade['mae_percent'] >= stop_loss_pct:
            simulated_pnl.append(-stop_loss_pct)
        elif trade['mfe_percent'] >= profit_target_pct:
            simulated_pnl.append(profit_target_pct)
        else:
            simulated_pnl.append(trade['captured_pct'])
            
    simulated_pnl = np.array(simulated_pnl)
    
    profit_factor = np.sum(simulated_pnl[simulated_pnl > 0]) / abs(np.sum(simulated_pnl[simulated_pnl < 0])) if np.sum(simulated_pnl < 0) != 0 else np.inf
    win_rate = np.sum(simulated_pnl > 0) / len(simulated_pnl) if len(simulated_pnl) > 0 else 0
    sharpe_ratio = calculate_sharpe_ratio(simulated_pnl)
    
    return profit_factor, win_rate * 100, sharpe_ratio

def analyze_trades(strategy_name):
    print(f"--- Starting VIX-Aware MAE/MFE Analysis for '{strategy_name}' ---")
    
    log_folder = os.path.join('backtest_logs', strategy_name)
    trade_files = glob.glob(os.path.join(log_folder, '*_trade_details.csv'))

    if not trade_files:
        print(f"Error: No trade logs found in '{log_folder}'.")
        return

    all_trades_df = pd.concat([pd.read_csv(f) for f in trade_files], ignore_index=True)
    
    required_cols = ['vix_on_entry', 'mae_percent', 'mfe_percent', 'captured_pct', 'entry_price', 'stop_loss']
    if not all(col in all_trades_df.columns for col in required_cols):
        print(f"Error: Trade log is missing required columns. Please use the latest simulator version.")
        print(f"Missing: {[col for col in required_cols if col not in all_trades_df.columns]}")
        return
        
    print(f"Loaded {len(all_trades_df)} total trade events.")

    all_trades_df['initial_risk_pct'] = ((all_trades_df['entry_price'] - all_trades_df['stop_loss']) / all_trades_df['entry_price']) * 100

    vix_buckets = {
        'Calm (VIX < 15)': (0, 15),
        'Moderate (VIX 15-22)': (15, 22),
        'High (VIX > 22)': (22, 999)
    }
    
    profit_target_multiplier_range = np.arange(1.0, 4.5, 0.5)

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
        
        best_sharpe = -np.inf
        best_params = {}

        avg_risk_in_bucket = bucket_df['initial_risk_pct'].mean()
        print(f"  - Average Initial Risk (Stop-Loss) in this bucket: {avg_risk_in_bucket:.2f}%")

        for rr in profit_target_multiplier_range:
            sim_stop_loss = avg_risk_in_bucket
            sim_profit_target = avg_risk_in_bucket * rr

            pf, wr, sharpe = run_what_if_simulation(bucket_df, sim_stop_loss, sim_profit_target)
            
            result = {
                'vix_regime': bucket_name,
                'avg_stop_loss_pct': sim_stop_loss,
                'profit_target_multiplier': rr,
                'sharpe_ratio': sharpe,
                'profit_factor': pf,
                'win_rate_pct': wr,
            }
            all_results.append(result)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = result
        
        if best_params:
            print("\n  Optimal Simulated Parameters Found (based on Sharpe Ratio):")
            print(f"  - Using Avg. SL of: {best_params['avg_stop_loss_pct']:.2f}%")
            print(f"  - Optimal Profit Target Multiplier: {best_params['profit_target_multiplier']:.1f}x")
            print(f"  - Resulting Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
            print(f"  - Resulting Profit Factor: {best_params['profit_factor']:.2f}")
            print(f"  - Resulting Win Rate: {best_params['win_rate_pct']:.1f}%")

    if not all_results:
        print("\nNo results generated.")
        return
        
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(log_folder, f'{timestamp}_daily_mae_mfe_analysis.csv')
    results_df.to_csv(output_filename, index=False, float_format='%.2f')
    
    print(f"\n--- Analysis Complete ---")
    print(f"Detailed simulation grid saved to: {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VIX-Aware MAE/MFE Analyzer for Daily TFL Strategy")
    parser.add_argument('--strategy', type=str, required=True, help='The name of the strategy folder in backtest_logs.')
    args = parser.parse_args()
    
    analyze_trades(args.strategy)
