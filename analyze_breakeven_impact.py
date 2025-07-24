# analyze_breakeven_impact.py
#
# Description:
# This script analyzes the output of the backtester to provide a quantitative
# assessment of the 'use_aggressive_breakeven' feature. It reads the most
# recent trades log, isolates the trades exited by the aggressive breakeven rule,
# and simulates what would have happened to them if the rule was not in place.
#
# This provides a data-driven way to determine if the feature is saving more
# capital than the potential profit it might be costing.

import pandas as pd
import os
import glob
import sys

# --- CONFIGURATION ---
config = {
    'log_folder': 'backtest_logs',
    'daily_data_folder': 'data/processed/daily'
}

def find_latest_trades_file(log_folder):
    """Finds the most recently created trades detail CSV file in the log folder."""
    try:
        list_of_files = glob.glob(os.path.join(log_folder, '*_trades_detail_*.csv'))
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        print(f"Error finding latest trades file: {e}")
        return None

def analyze_breakeven_impact(cfg):
    """
    Main function to run the analysis.
    """
    print("--- Aggressive Breakeven Impact Analysis ---")

    # 1. Find and load the latest trades log
    latest_trades_file = find_latest_trades_file(cfg['log_folder'])
    if not latest_trades_file:
        print(f"Error: No trade detail files found in '{cfg['log_folder']}'.")
        print("Please run a backtest with 'use_aggressive_breakeven' enabled first.")
        sys.exit()
    
    print(f"Analyzing trades from: {os.path.basename(latest_trades_file)}")
    trades_df = pd.read_csv(latest_trades_file, parse_dates=['entry_date', 'exit_date'])

    # 2. Isolate the relevant trades
    breakeven_trades = trades_df[trades_df['exit_type'] == 'Aggressive Breakeven Stop'].copy()

    if breakeven_trades.empty:
        print("\nNo trades were exited by the 'Aggressive Breakeven Stop' rule in this log.")
        print("Analysis complete.")
        return

    print(f"Found {len(breakeven_trades)} trades exited by the aggressive breakeven rule.")

    # 3. Load daily data for the symbols involved
    print("Loading required daily price data...")
    daily_data = {}
    symbols_needed = breakeven_trades['symbol'].unique()
    for symbol in symbols_needed:
        try:
            file_path = os.path.join(cfg['daily_data_folder'], f"{symbol}_daily_with_indicators.csv")
            daily_data[symbol] = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
        except FileNotFoundError:
            print(f"  > Warning: Daily data for {symbol} not found. Some trades may be skipped.")
            continue
    
    # 4. Simulate the "what if" scenario for each trade
    print("Simulating alternative outcomes...")
    results = []
    for index, trade in breakeven_trades.iterrows():
        symbol = trade['symbol']
        if symbol not in daily_data:
            continue

        df_daily = daily_data[symbol]
        
        # Define the original parameters of the trade
        original_stop_loss = trade['initial_stop_loss']
        profit_target = trade['target']
        exit_date = trade['exit_date']
        
        # Get the price data for the period after the premature exit
        future_candles = df_daily.loc[exit_date:].iloc[1:] # Start from the day after exit

        outcome = 'No Event'
        pnl_impact = 0

        for date, candle in future_candles.iterrows():
            # Check if the profit target would have been hit
            if candle['high'] >= profit_target:
                outcome = 'Missed Profit'
                # Calculate the P&L of the first leg that was missed
                missed_pnl = (profit_target - trade['entry_price']) * (trade['initial_shares'] / 2)
                # Subtract the small profit that was actually taken
                actual_pnl = trade['pnl']
                pnl_impact = actual_pnl - missed_pnl # This will be a negative number
                break
            
            # Check if the original, wider stop-loss would have been hit
            if candle['low'] <= original_stop_loss:
                outcome = 'Saved Loss'
                # Calculate the P&L that would have occurred
                hypothetical_loss = (original_stop_loss - trade['entry_price']) * trade['initial_shares']
                # Subtract the small profit that was actually taken
                actual_pnl = trade['pnl']
                pnl_impact = actual_pnl - hypothetical_loss # This will be a positive number
                break
        
        results.append({
            'symbol': symbol,
            'exit_date': exit_date,
            'outcome': outcome,
            'pnl_impact': pnl_impact
        })

    # 5. Aggregate and present the final stats
    print("\n--- ANALYSIS COMPLETE ---")
    results_df = pd.DataFrame(results)

    saved_trades = results_df[results_df['outcome'] == 'Saved Loss']
    missed_trades = results_df[results_df['outcome'] == 'Missed Profit']
    no_event_trades = results_df[results_df['outcome'] == 'No Event']

    total_pnl_saved = saved_trades['pnl_impact'].sum()
    total_pnl_missed = missed_trades['pnl_impact'].sum() # This is a negative value
    net_impact = total_pnl_saved + total_pnl_missed

    print("\n--- Summary Report ---")
    print(f"Total Trades Analyzed: {len(results_df)}")
    print("-" * 25)
    print(f"Trades where rule PREVENTED a larger loss: {len(saved_trades)}")
    print(f"  > Total P&L Saved by the rule: {total_pnl_saved:,.2f}")
    print(f"\nTrades where rule EXITED a winner prematurely: {len(missed_trades)}")
    print(f"  > Total Potential P&L Missed by the rule: {abs(total_pnl_missed):,.2f}")
    print(f"\nTrades with no significant event after exit: {len(no_event_trades)}")
    print("-" * 25)
    print(f"NET IMPACT OF AGGRESSIVE BREAKEVEN RULE: {net_impact:,.2f}")
    
    if net_impact > 0:
        print("\nConclusion: The rule is beneficial. It saved more money than it cost in missed profits.")
    else:
        print("\nConclusion: The rule is detrimental. It cost more in missed profits than it saved from losses.")


if __name__ == "__main__":
    analyze_breakeven_impact(config)
