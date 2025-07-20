Project Context & Implementation Details for LLM Analysis
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. The goal is for an LLM to fully understand the project's architecture, data flow, and the precise logic of the trading strategy to assist with future modifications, bug fixes, or the addition of new features.

1. High-Level Project Goal
The primary objective is to develop, backtest, and refine a profitable, long-only, pullback-based swing trading strategy. The strategy is applied to the universe of stocks within India's Nifty 200 index. The project consists of a full pipeline of scripts for data acquisition, data processing across multiple timeframes, and robust, event-driven backtesting.

2. System Architecture and Workflow
The project is a modular system composed of several Python scripts that work in sequence:

fyers_equity_scraper.py & fyers_nifty200_index_scraper.py: These scripts are responsible for connecting to the Fyers API to download and maintain a local database of daily historical stock and index data.

calculate_indicators.py: This script processes the raw daily data. It first resamples the data into multiple timeframes (daily, 2-day, weekly, monthly) and then calculates and adds a suite of technical indicators (ema_30, ema_50, atr_14, atr_ma_30) to each data file. The outputs are saved into separate directories for each timeframe (e.g., data/processed/daily/).

final_backtester.py: This is the core backtesting engine. It simulates the trading strategy on a user-selected timeframe (daily, 2day, etc.) using the corresponding processed data files. It generates two key output files: a summary performance report and a detailed log of every trade taken.

3. Core Trading Strategy: A Detailed Breakdown
The strategy is executed by the final_backtester.py script. The logic is event-driven and proceeds day by day.

A. Universe and Timeframe:

Universe: Nifty 200 stocks, as defined by the list of symbols in nifty200.csv.

Timeframe: The simulation can be run on daily, 2day, weekly, or monthly timeframes by changing the timeframe parameter in the script's configuration.

B. Entry Filters (Applied Sequentially):

Market Regime Filter (Master Switch):

Purpose: Prevents new long positions during a market downturn.

Data Source: Uses the daily NIFTY200_INDEX_daily_with_indicators.csv file, regardless of the strategy's chosen timeframe.

Logic: On any given day, if the Nifty 200 index's closing price is greater than its 50-day EMA (ema_50), the market is considered in an "uptrend," and the script proceeds. Otherwise, the entry logic is skipped for that day.

Volatility Normalization (ATR) Filter:

Purpose: Avoids entering trades during periods of excessively high volatility.

Logic: On the entry trigger day (Day T), the script checks if the stock's 14-day ATR (atr_14) is greater than its 30-day moving average of ATR (atr_ma_30) multiplied by a factor (e.g., 1.4). If volatility is too high, the trade is rejected. (Note: This filter is currently implemented but disabled by default ('atr_filter': False) in the configuration).

Volume Confirmation Filter:

Purpose: Ensures that the entry breakout is supported by significant buying interest.

Logic: On the entry trigger day (Day T), the script checks if the stock's volume is greater than its 20-day average volume multiplied by a factor (e.g., 1.3). If the volume is insufficient, the trade is rejected.

C. Entry Logic (A Two-Day Process):

Day T-1 (The Setup Day): The script scans every stock at the close of the market to find a specific 5-point pattern. All conditions must be met:

The candle on Day T-1 must be green (close > open).

The close of this green candle must be in the upper 50% of the candle's total range.

The close of the green candle must be above the stock's 30-period EMA.

It must be preceded by one or more consecutive red candles.

Day T (The Trigger Day): If a valid setup was confirmed on Day T-1, the script proceeds to Day T with a potential trade order.

Trigger Price Calculation: The entry trigger price is the absolute highest high of the entire setup pattern.

Execution Condition: A trade is entered on Day T if the high of Day T is greater than or equal to the trigger price, AND the open of Day T is less than the trigger price.

D. Position Sizing and Risk Management:

Risk Allocation: Risks a fixed percentage (e.g., 3.0%) of the current total portfolio equity on each trade.

Equity Calculation Timing: Portfolio equity is updated after all exits for the day but before new entries are scanned, ensuring risk is based on current capital.

Initial Stop-Loss: Set at the lowest low of the 5 trading days immediately preceding the entry day.

E. Trade Management and Exit Logic (Two-Leg Strategy):

Leg 1 - Partial Profit Exit:

A take-profit target is set at a 1:1 risk/reward ratio.

If the price hits this target, 50% of the position is sold.

The stop-loss for the remaining 50% is immediately moved to the entry price (breakeven).

Leg 2 - Trailing Stop-Loss (Active from Day 1):

This logic runs daily for any open position that is currently profitable.

Breakeven Rule: The stop-loss is first moved to the higher of its current level or the breakeven price.

Trailing Rule: If the current day's candle is green, the stop-loss is then moved to the higher of its current level or the low of that green candle.

Final Exit: The entire remaining position is closed if the price hits the current stop-loss level.

4. Data and File System Structure
Input Data Directory: Raw data is in historical_data/. Processed data is in data/processed/, subdivided by timeframe (e.g., data/processed/daily/).

Stock Data Filename Convention: {SYMBOL}_daily_with_indicators.csv.

Index Data Filename: NIFTY200_INDEX_daily_with_indicators.csv.

Output Directory: All results are saved to the backtest_logs/ directory.

Output Files: The script generates a .txt summary report and a .csv detailed trade log, both prefixed with a timestamp. The trade log includes granular details like portfolio_equity_on_entry, risk_per_share, and initial_shares for auditing.