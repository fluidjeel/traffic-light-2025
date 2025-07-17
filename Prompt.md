Project Brief: Create an Accurate Python Portfolio Backtester
Objective:
Your task is to write a single, robust Python script named final_backtester.py. This script will perform a portfolio-level backtest of a specific trading strategy on a list of stocks from the Indian Nifty 200 index. The absolute highest priority is the accuracy of the portfolio simulation and the final CAGR calculation.

1. Critical Context & History (What Went Wrong Before)
Previous attempts to build this script failed for two main reasons. You must understand and avoid these pitfalls:

The "Slowdown" Bug: An early version processed a single, massive dataframe day-by-day. Inside the daily loop, it would re-scan the entire dataframe to find historical data for a stock. This was extremely inefficient and made the script "hang."

Solution: The script should pre-process the data by grouping it into a dictionary of dataframes (one for each stock symbol) before the main simulation loop begins. This allows for instant lookups.

The "Unrealistic CAGR" Bug: A later version had a critical flaw in its portfolio logic. It would update the main portfolio equity immediately after a partial profit exit and then use that newly inflated equity to size new trades entered on the same day. This is called intra-day compounding and is unrealistic. It led to an impossibly high CAGR of over 400%.

Solution: The simulation engine must operate on a strict day-by-day basis. The portfolio equity value must be fixed at the start of each day. All calculations for position sizing on a given day must use this start-of-day equity. P&L from all of that day's exits should be summed up and added to the portfolio equity only once, at the very end of the day.

2. Input Files & Structure
The script must use the following files, which will be located in the same directory as the script itself:

nifty200.csv:

A CSV file containing the list of stock symbols to be tested.

It has a single header named Symbol.

daily_with_indicators/ (Folder):

This folder contains the historical data for each stock.

The filename format is [SYMBOL]_daily_with_indicators.csv (e.g., RELIANCE_daily_with_indicators.csv).

Required Columns in each file: datetime, open, high, low, close, and EMA_30. The script should be robust enough to handle variations in capitalization (e.g., ema_30).

3. The Trading Strategy (The "simplified_ts" Logic)
This is the exact, non-negotiable logic for the trading strategy.

A. Entry Rules:
Signal Condition: The script must identify a two-candle pattern on the daily chart:

Candle 1 (T-2 days) must be a red candle (close < open).

Candle 2 (T-1 days) must be a green candle (close > open).

The close of the green candle (T-1) must be above the 30-day EMA (ema_30).

Entry Trigger (on Day T): If the signal is present, a trade is entered on the following day (Day T) only if:

The open price of Day T is below the entry_trigger_price.

The high of Day T is greater than or equal to the entry_trigger_price.

The entry_trigger_price is defined as the max(high of T-2, high of T-1).

The actual entry_price for the trade is the entry_trigger_price.

B. Risk Management & Position Sizing:
Initial Stop-Loss: The stop-loss for the trade is set at the lowest low of the 5 candles immediately preceding the entry day (i.e., days T-5 through T-1).

Position Sizing (CRITICAL):

The number of shares for a trade is calculated based on a fixed risk percentage of the portfolio.

risk_per_share = entry_price - stop_loss

capital_to_risk = equity_at_start_of_day * (risk_per_trade_percent / 100)

shares_to_buy = floor(capital_to_risk / risk_per_share)

This shares_to_buy value is fixed for the entire duration of the trade. It does not change after a partial exit.

C. Exit Rules (Two-Stage):
Stage 1: Partial Profit Exit

A 1:1 Risk/Reward target is calculated: target_price_1_1 = entry_price + risk_per_share.

If the high of any day touches or exceeds this target, 50% of the initial shares are sold at the target_price_1_1.

This can only happen once per trade.

Stage 2: Trailing Stop-Loss for Remainder

This logic applies to the entire position from the day of entry.

At the end of each day, if the position is profitable (current_close > entry_price) AND the day's candle was green (close > open), the stop-loss is updated.

The new stop-loss becomes: max(current_stop_loss, low_of_the_green_candle). The stop-loss can only move up.

The trade is finally closed (the remaining 50% is sold) when the price hits this trailed stop-loss.

4. Script Requirements & Output
File Name: final_backtester.py

Configuration: Key parameters (INITIAL_CAPITAL, RISK_PER_TRADE_PERCENT, DATA_FOLDER, etc.) should be at the top of the script for easy modification.

Logging & Output:

The script must create a folder named backtest_logs if it doesn't exist.

It must save two timestamped files inside this folder for each run:

[timestamp]_summary_report.txt: A text file containing the final portfolio summary (Initial/Final Capital, Net P&L, Total Return %, CAGR, and Backtest Period in years).

[timestamp]_trades_log.csv: A CSV file detailing every single exit transaction with columns: symbol, entry_date, exit_date, pnl, and exit_type (e.g., 'Stop-Loss', 'Partial Profit (1:1)').

Console Output: The script should print its progress continuously (e.g., "Processing Day X/Y...") so the user knows it has not stalled.

Please generate the Python code that fulfills all of these requirements, paying special attention to the historical context to avoid repeating past mistakes.