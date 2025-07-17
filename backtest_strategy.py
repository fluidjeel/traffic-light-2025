# backtest_strategy.py
#
# Description:
# This script backtests the defined trading strategy for a single stock.
# This is a complete rewrite with a more robust, state-based logic to ensure
# accuracy of trade entries and exits.
#
# Prerequisites:
# 1. You must have run the previous scripts to generate the data folders:
#    - 'daily_with_indicators'
# 2. Required libraries installed: pip install pandas
#
# How to use:
# 1. Set the `STOCK_TO_BACKTEST` variable to the desired symbol.
# 2. Run this script. It will print a detailed log of trades and a final performance summary.

import os
import pandas as pd
from datetime import datetime

# --- Backtest Configuration ---
STOCK_TO_BACKTEST = "RELIANCE"  # Change this to the stock symbol you want to test
START_DATE = "2023-01-01"
STOP_LOSS_BUFFER_PERCENT = 0.010
RISK_REWARD_RATIO = 1.0

# --- Strategy Functions ---
def check_daily_entry_setup(df, current_day_index, ema_period):
    """
    Checks if a valid entry setup was confirmed at the close of the PREVIOUS day.
    """
    ema_column = f'EMA_{ema_period}'
    # A setup is confirmed at the close of the 'confirmation_candle' (yesterday)
    # The pattern is based on the 'confirmation_candle' and the 'previous_candle' (day before yesterday)
    if current_day_index < 6 or ema_column not in df.columns:
        return None
    
    confirmation_candle = df.iloc[current_day_index - 1]
    previous_candle = df.iloc[current_day_index - 2]

    is_previous_candle_red = previous_candle['close'] < previous_candle['open']
    is_confirmation_candle_green = confirmation_candle['close'] > confirmation_candle['open']
    is_price_above_ema = confirmation_candle['close'] > confirmation_candle[ema_column]

    if is_previous_candle_red and is_confirmation_candle_green and is_price_above_ema:
        entry_price = max(confirmation_candle['high'], previous_candle['high'])
        lowest_low_5 = df['low'].iloc[current_day_index-6:current_day_index-1].min()
        stop_loss = lowest_low_5 * (1 - STOP_LOSS_BUFFER_PERCENT)
        return {"setup_type": "Daily", "entry_price": entry_price, "stop_loss": stop_loss}
    
    return None

def backtest_stock(symbol):
    """
    Main function to run the backtest for a single stock.
    """
    print(f"--- Starting Backtest for {symbol} from {START_DATE} ---")

    # 1. Load and Prepare Data
    try:
        daily_df = pd.read_csv(f"daily_with_indicators/{symbol}_daily_with_indicators.csv", index_col='datetime', parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files for {symbol}. {e}")
        return

    df = daily_df[daily_df.index >= START_DATE]
    if df.empty:
        print("No data available for the specified date range.")
        return

    trades = []
    state = 'SCANNING'  # Possible states: SCANNING, PENDING_ENTRY, IN_TRADE_FULL, IN_TRADE_HALF
    trade_details = {}
    
    # 2. Main Backtesting Loop
    # We start from index 2 to have enough history for the setup check
    for i in range(2, len(df)):
        current_candle = df.iloc[i]
        current_date = df.index[i].date()

        # --- State: IN_TRADE (FULL or HALF position) ---
        if state in ['IN_TRADE_FULL', 'IN_TRADE_HALF']:
            # Check for stop-loss hit
            if current_candle['low'] <= trade_details['stop_loss']:
                print(f"  > {current_date}: STOP-LOSS HIT at {trade_details['stop_loss']:.2f}")
                trade_details['exit_date'] = current_date
                trade_details['exit_price'] = trade_details['stop_loss']
                trades.append(trade_details)
                state = 'SCANNING'
                trade_details = {}
                continue

            # Check for 1:1 target hit (only for full positions)
            if state == 'IN_TRADE_FULL' and current_candle['high'] >= trade_details['target_price']:
                print(f"  > {current_date}: 1:1 TARGET HIT at {trade_details['target_price']:.2f}. Selling 50%.")
                state = 'IN_TRADE_HALF'
                trade_details['stop_loss'] = trade_details['entry_price'] # Move SL to breakeven
                print(f"  > New Stop-Loss (Breakeven): {trade_details['stop_loss']:.2f}")

            # Check for reverse signal exit (only for half positions)
            if state == 'IN_TRADE_HALF':
                prev_candle = df.iloc[i-1]
                if prev_candle['close'] > prev_candle['open'] and current_candle['close'] < current_candle['open']:
                    if current_candle['low'] < prev_candle['low']:
                        print(f"  > {current_date}: REVERSE SIGNAL EXIT at {prev_candle['low']:.2f}")
                        trade_details['exit_date'] = current_date
                        trade_details['exit_price'] = prev_candle['low']
                        trades.append(trade_details)
                        state = 'SCANNING'
                        trade_details = {}
                        continue
            
            # Trailing Stop-Loss Logic (at the end of the day)
            if current_candle['close'] > trade_details['entry_price']:
                last_green_candle = df.iloc[:i+1][df.iloc[:i+1]['close'] > df.iloc[:i+1]['open']].iloc[-1]
                new_sl = last_green_candle['low'] * (1 - STOP_LOSS_BUFFER_PERCENT)
                if new_sl > trade_details['stop_loss']:
                    print(f"  > {current_date}: TRAILING STOP-LOSS updated from {trade_details['stop_loss']:.2f} to {new_sl:.2f}")
                    trade_details['stop_loss'] = new_sl

        # --- State: PENDING_ENTRY ---
        elif state == 'PENDING_ENTRY':
            if current_candle['high'] >= trade_details['entry_price']:
                print(f"\n*** NEW TRADE on {current_date} ***")
                print(f"  > Pending order filled for {trade_details['setup_type']} setup.")
                
                state = 'IN_TRADE_FULL'
                trade_details['entry_date'] = current_date
                risk = trade_details['entry_price'] - trade_details['stop_loss']
                trade_details['target_price'] = trade_details['entry_price'] + (risk * RISK_REWARD_RATIO)
                
                print(f"  > Entry: {trade_details['entry_price']:.2f}, SL: {trade_details['stop_loss']:.2f}, Target: {trade_details['target_price']:.2f}")
            else:
                print(f"  > {current_date}: Pending order for {trade_details['setup_type']} NOT FILLED. Order cancelled.")
                state = 'SCANNING'
                trade_details = {}
        
        # --- State: SCANNING ---
        elif state == 'SCANNING':
            setup = check_daily_entry_setup(df, i, 30)
            if setup:
                state = 'PENDING_ENTRY'
                trade_details = setup # Store the setup details
                print(f"\n  > {df.index[i-1].date()}: {setup['setup_type']} setup CONFIRMED. Pending order placed for {current_date}.")
                print(f"    > Entry: {setup['entry_price']:.2f}, SL: {setup['stop_loss']:.2f}")

    # 3. Print Results
    print(f"\n--- Backtest for {symbol} Complete ---")
    if not trades:
        print("No trades were executed.")
        return

    trade_log = pd.DataFrame(trades)
    trade_log['pnl'] = trade_log['exit_price'] - trade_log['entry_price']
    
    print("\n--- Trade Log ---")
    print(trade_log[['setup_type', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl']])

    print("\n--- Performance Summary ---")
    print(f"Total Trades: {len(trade_log)}")
    print(f"Gross P/L: {trade_log['pnl'].sum():.2f}")
    
    wins = trade_log[trade_log['pnl'] > 0]
    losses = trade_log[trade_log['pnl'] <= 0]
    
    print(f"Winning Trades: {len(wins)}")
    print(f"Losing Trades: {len(losses)}")
    if len(trade_log) > 0:
        win_rate = (len(wins) / len(trade_log)) * 100
        print(f"Win Rate: {win_rate:.2f}%")
    if len(wins) > 0:
        print(f"Average Win: {wins['pnl'].mean():.2f}")
    if len(losses) > 0:
        print(f"Average Loss: {losses['pnl'].mean():.2f}")


if __name__ == "__main__":
    backtest_stock(STOCK_TO_BACKTEST)
