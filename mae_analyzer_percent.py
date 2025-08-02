# mae_analyzer_percent.py
#
# Description:
# An enhanced analysis tool tailored for strategies using a PERCENT-based stop-loss.
# It provides deep insights into stop-loss and profit-taking efficiency by analyzing
# MAE and MFE from trade logs.
#
# MODIFICATION (v2.0):
# - Added a new "Actionable Trade Insights" section to pinpoint specific trades
#   of interest (e.g., winners with high drawdown).
# - The source log file for each trade is now loaded and can be displayed.
#
# Features:
# - MFE Analysis: Measures profit-taking efficiency against peak potential profit.
# - Regime-Aware Analysis: Segments analysis by VIX levels.
# - Percentage Stop-Loss Simulation: Runs "what-if" scenarios to find the
#   optimal fixed percentage stop-loss for the strategy.

import pandas as pd
import numpy as np
import os
import argparse
import glob

# --- Analysis Configuration ---
PERCENT_STOPS = np.arange(3.0, 15.1, 1.0)  # Test stops from 3% to 15% in 1% increments
VIX_REGIME_THRESHOLD = 22 # VIX level to distinguish between calm and volatile markets
SAFE_ZONE_PERCENTILE = 0.95 # Preserves 95% of original winning trades in recommendations
INSIGHT_TRADE_COUNT = 3 # Number of trades to show in the actionable insights section

def analyze_excursions(trades_df, regime_name="Overall"):
    """
    Performs a full MAE and MFE analysis on a given set of trades.
    """
    if trades_df.empty:
        print(f"\n--- No trades to analyze for {regime_name} Regime ---")
        return

    print(f"\n{'='*25} EXCURSION ANALYSIS: {regime_name.upper()} REGIME {'='*25}")
    
    winners = trades_df[trades_df['pnl'] > 0].copy()
    losers = trades_df[trades_df['pnl'] <= 0].copy()

    # --- MFE Analysis (Profit-Taking Efficiency) ---
    print("\n[ MFE ANALYSIS (Profit-Taking Efficiency) ]")
    if not winners.empty and 'mfe_percent' in winners.columns:
        winners['efficiency'] = (winners['captured_pct'] / winners['mfe_percent']).replace([np.inf, -np.inf], 0)
        avg_mfe_winners = winners['mfe_percent'].mean()
        avg_captured = winners['captured_pct'].mean()
        avg_efficiency = winners['efficiency'].mean()
        print(f"  - Winning Trades Avg. Peak Profit (MFE): {avg_mfe_winners:.2f}%")
        print(f"  - Winning Trades Avg. Captured Profit:   {avg_captured:.2f}%")
        print(f"  - Average Profit Capture Efficiency:     {avg_efficiency:.1%}")
    else:
        print("  - No winning trades with MFE data to analyze.")

    if not losers.empty and 'mfe_percent' in losers.columns:
        avg_mfe_losers = losers['mfe_percent'].mean()
        print(f"  - Losing Trades Avg. Peak Profit (MFE):  {avg_mfe_losers:.2f}% (before turning to loss)")
    else:
        print("  - No losing trades with MFE data to analyze.")

    # --- MAE Analysis (Stop-Loss Optimization) ---
    print("\n[ MAE ANALYSIS (Stop-Loss Optimization) ]")
    if not winners.empty and 'mae_percent' in winners.columns:
        print(f"  - Winning Trades Avg. Drawdown (MAE):    {winners['mae_percent'].mean():.2f}%")
        safe_stop_level = winners['mae_percent'].quantile(1 - SAFE_ZONE_PERCENTILE)
        print(f"  - Recommended 'Safe Zone' Stop:        <{safe_stop_level:.2f}% (preserves {SAFE_ZONE_PERCENTILE:.0%} of winners)")
    else:
        print("  - No winning trades with MAE data to analyze.")
        
    if not losers.empty and 'mae_percent' in losers.columns:
        print(f"  - Losing Trades Avg. Drawdown (MAE):     {losers['mae_percent'].mean():.2f}%")
    else:
        print("  - No losing trades with MAE data to analyze.")

    # --- NEW: Actionable Trade Insights ---
    print("\n[ ACTIONABLE TRADE INSIGHTS ]")
    if not winners.empty and 'mae_percent' in winners.columns:
        # Winners with the most drawdown (good trades that almost got stopped out)
        high_mae_winners = winners.sort_values(by='mae_percent', ascending=False).head(INSIGHT_TRADE_COUNT)
        print("  - Top Winning Trades with Highest Drawdown (MAE):")
        for _, trade in high_mae_winners.iterrows():
            print(f"    - ID: {trade['setup_id']}, MAE: {trade['mae_percent']:.2f}%, File: {trade['source_file']}")

        # Winners with the worst profit capture (left the most money on the table)
        low_efficiency_winners = winners.sort_values(by='efficiency', ascending=True).head(INSIGHT_TRADE_COUNT)
        print("\n  - Top Winning Trades with Lowest Profit Capture Efficiency:")
        for _, trade in low_efficiency_winners.iterrows():
            print(f"    - ID: {trade['setup_id']}, Captured: {trade['captured_pct']:.2f}% of {trade['mfe_percent']:.2f}%, File: {trade['source_file']}")
    else:
        print("  - No winning trades to generate insights from.")


def simulate_percent_stops(trades_df, regime_name="Overall"):
    """
    Simulates the outcome of trades using a range of fixed percentage stops.
    """
    if trades_df.empty or 'mae_percent' not in trades_df.columns:
        print(f"\n--- Percentage Stop Simulation skipped for {regime_name} (Not enough data) ---")
        return
        
    print(f"\n{'='*25} FIXED PERCENTAGE STOP-LOSS SIMULATION: {regime_name.upper()} REGIME {'='*25}")
    print("\n  Stop-Loss % | Win Rate | Profit Factor | Trades Kept")
    print("  -------------------------------------------------------")

    for stop_pct in PERCENT_STOPS:
        # A trade is "stopped out" if its actual MAE in percent exceeds the simulated stop level.
        kept_trades = trades_df[trades_df['mae_percent'] < stop_pct]
        
        if kept_trades.empty:
            continue

        win_rate = (kept_trades['pnl'] > 0).sum() / len(kept_trades) * 100
        gross_profit = kept_trades[kept_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(kept_trades[kept_trades['pnl'] <= 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        print(f"  {stop_pct:^11.1f} | {win_rate:^8.1f}% | {profit_factor:^13.2f} | {len(kept_trades):^11} / {len(trades_df)}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced MAE/MFE Analyzer for percent-based stop-loss strategies.")
    parser.add_argument('--strategy', type=str, required=True, help="The name of the strategy folder in 'backtest_logs' (e.g., 'simulator_monthly_advanced').")
    args = parser.parse_args()

    log_dir = os.path.join('backtest_logs', args.strategy)
    if not os.path.isdir(log_dir):
        print(f"Error: Log directory not found at '{log_dir}'")
        return

    all_trade_files = glob.glob(os.path.join(log_dir, '*_trade_details.csv'))
    if not all_trade_files:
        print(f"No trade logs found in '{log_dir}'.")
        return

    # --- MODIFIED: Load data and add source file information ---
    all_dfs = []
    for f in all_trade_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = os.path.basename(f)
            all_dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty log file: {f}")
            continue
            
    if not all_dfs:
        print("All found trade logs were empty.")
        return
        
    all_trades_df = pd.concat(all_dfs, ignore_index=True)
    # --- END MODIFICATION ---
    
    print(f"--- Loaded {len(all_trades_df)} trades from {len(all_trade_files)} log files for strategy '{args.strategy}' ---")

    if 'vix_close' in all_trades_df.columns:
        high_vix_trades = all_trades_df[all_trades_df['vix_close'] > VIX_REGIME_THRESHOLD]
        low_vix_trades = all_trades_df[all_trades_df['vix_close'] <= VIX_REGIME_THRESHOLD]
        
        analyze_excursions(high_vix_trades, "High VIX")
        simulate_percent_stops(high_vix_trades, "High VIX")
        
        analyze_excursions(low_vix_trades, "Low VIX")
        simulate_percent_stops(low_vix_trades, "Low VIX")
    else:
        print("\nWarning: 'vix_close' column not found. Skipping regime-based analysis.")

    analyze_excursions(all_trades_df, "Overall")
    simulate_percent_stops(all_trades_df, "Overall")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
