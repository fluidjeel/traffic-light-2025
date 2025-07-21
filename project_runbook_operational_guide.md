Project Runbook & Operational Guide
Version: 3.0
Last Updated: 2025-07-21
Strategy: Nifty 200 Pullback System (Standard & Immediate Entry)

1. Project Overview
1.1. Goal:
This project is a complete algorithmic trading system designed to backtest and potentially automate a long-only, pullback-based swing trading strategy on the Nifty 200 stock universe. It includes a full pipeline for data acquisition, multi-timeframe indicator processing, and two distinct backtesting engines for different entry logics.

1.2. Core Philosophy:
The strategy aims to identify established uptrends that are undergoing a brief, orderly pullback. It then seeks to enter a trade at the moment the trend appears to be resuming, confirmed by multiple filters including market direction, volume, and relative strength.

2. System Architecture & Script Dependencies
The project operates as a sequential pipeline. Each script must be run in the correct order as the output of one script is the input for the next.

Execution Order:

fyers_equity_scraper.py & fyers_nifty200_index_scraper.py (Data Acquisition)

calculate_indicators.py (Data Processing)

final_backtester.py OR final_backtester_immediate.py (Strategy Simulation)

analyze_missed_trades.py (Post-Hoc Analysis)

2.1. Data Acquisition Scripts:

Purpose: To download and maintain a local database of raw, daily OHLCV (Open, High, Low, Close, Volume) data from the Fyers API.

fyers_equity_scraper.py: Downloads data for all individual stocks listed in nifty200.csv.

fyers_nifty200_index_scraper.py: A specialized version to download data for the Nifty 200 index itself, which is critical for the system's filters.

Key Feature: These scripts perform incremental updates. They check for existing data and only download new data since the last run, making daily updates very fast and efficient.

2.2. calculate_indicators.py:

Purpose: To transform the raw daily data into processed, indicator-rich files ready for the backtesters.

Process:

It reads the raw daily data for each stock and the index.

It resamples this daily data into four distinct timeframes: daily, 2day, weekly, and monthly.

For each timeframe, it calculates a suite of technical indicators (ema_30, ema_50, atr_14, atr_ma_30).

It saves the processed files into separate, clearly named directories (e.g., data/processed/daily/, data/processed/weekly/).

2.3. Backtesting Engines:

final_backtester.py (Standard Strategy):

Purpose: To simulate the strategy using a traditional, end-of-period logic.

Timeframes: daily, 2day, weekly, monthly.

Core Logic: It makes all entry decisions based on the pattern of completed candles. For example, when running on a weekly timeframe, it will only check for a new entry signal at the close of the market on Friday, based on the shape of that week's completed candle.

final_backtester_immediate.py (Fast Entry Strategy):

Purpose: To simulate an accelerated entry strategy for higher timeframes.

Timeframes: weekly-immediate, monthly-immediate.

Core Logic: This engine uses a hybrid model. It identifies the setup pattern on the completed higher-timeframe charts (e.g., last week's candles) but then switches to a daily scan to look for an entry trigger during the formation of the current week's candle. This allows for a potentially faster entry.

2.4. analyze_missed_trades.py:

Purpose: To provide insight into the opportunity cost of capital constraints.

Process: After a backtest is complete, this script reads the missed_trades_log.csv file. It then runs a simplified, non-compounding simulation on these trades to determine what their performance would have been if capital had been available. It reports the potential win rate and total return of these missed opportunities.

3. Detailed Strategy Routines & Logic
(This section details the logic universally applied by both backtesters unless specified otherwise)

3.1. Entry Filters (The Gauntlet):
A potential trade setup must pass all of the following filters in sequence to be considered for entry.

Market Regime Filter:

Question: Is the overall market in an uptrend?

Logic: Checks if the daily Nifty 200 index close is above its 50-day EMA. If not, no new long trades are considered for that day.

Relative Strength Filter:

Question: Is this stock stronger than the overall market?

Logic: Checks if the stock's 30-day percentage return is greater than the Nifty 200's 30-day return.

Volume Filter:

Question: Is there conviction behind this breakout?

Logic: Checks if the volume on the entry day is at least 1.3 times its 20-day average.

ATR Filter (Disabled by default):

Question: Is the stock's volatility stable?

Logic: Checks if the short-term volatility (14-day ATR) is not excessively higher than its long-term average (30-day MA of ATR).

3.2. Core Entry Pattern (Two-Day Process):

Day T-1 (The Setup Day):

The candle must be green.

The close must be in the upper 50% of the candle's range.

The close must be above the 30-period EMA.

It must be preceded by one or more consecutive red candles.

Day T (The Trigger Day):

Trigger Price: The highest high of the entire setup pattern (red candles + green candle).

Execution: A trade is entered if the high of Day T is >= the Trigger Price AND the open of Day T is < the Trigger Price.

3.3. Position Sizing & Risk:

Risk Allocation: A fixed percentage of the current total portfolio equity is risked on each trade.

Equity Calculation: Portfolio equity is recalculated after all exits for the day are processed but before any new entries are considered.

Initial Stop-Loss: The lowest low of the 5 trading days preceding the entry day.

3.4. Trade Management & Exits (Two-Leg Strategy):

Leg 1 (Partial Profit): 50% of the position is sold when the price hits a 1:1 risk/reward target. The stop-loss for the remaining shares is immediately moved to the entry price (breakeven).

Leg 2 (Trailing Stop): For any profitable trade, the stop-loss is trailed daily. The new stop is set to the higher of either the breakeven price or the low of the most recent green candle.

Final Exit: The entire remaining position is closed if the price hits the current stop-loss level.

4. Operational Oversight Checklist
Daily (5-10 mins):

[ ] Confirm data scrapers and backtester scripts ran without errors.

[ ] If live trading, reconcile trades in the log with orders in the broker terminal.

[ ] Check financial news for corporate actions (splits, etc.) for any open positions.

Weekly (15-20 mins):

[ ] Manually audit one winning and one losing trade from the week against the strategy rules.

[ ] Review the weekly performance. Is it in line with backtested expectations?

[ ] Check for any changes to the Nifty 200 index constituents.

Monthly (30-45 mins):

[ ] Aggregate and review the month's performance metrics (Win Rate, Profit Factor, etc.).

[ ] Analyze the monthly drawdown. Was it within historical norms?

[ ] Run analyze_missed_trades.py to understand the opportunity cost of the current capital allocation.