Project Context & Implementation Details for LLM Analysis
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. The goal is for an LLM to fully understand the project's architecture, data flow, and the precise logic of the two distinct trading strategies to assist with future modifications, bug fixes, or the addition of new features.

1. High-Level Project Goal
The primary objective is to develop, backtest, and refine a profitable, long-only, pullback-based swing trading strategy. The strategy is applied to the universe of stocks within India's Nifty 200 index. The project consists of a full pipeline of scripts for data acquisition, data processing across multiple timeframes, and two separate, robust, event-driven backtesting engines for standard and accelerated entries.

2. System Architecture and Workflow
The project is a modular system composed of several Python scripts that work in sequence:

Data Acquisition (fyers_..._scraper.py scripts): These scripts connect to the Fyers API to download and maintain a local database of daily historical stock and index data.

Indicator Calculation (calculate_indicators.py): This script processes the raw daily data. It first resamples the data into multiple timeframes (daily, 2-day, weekly, monthly) and then calculates and adds the necessary technical indicators (ema_30, ema_50, atr_14, atr_ma_30, etc.) to each data file. The outputs are saved into separate directories for each timeframe (e.g., data/processed/daily/).

Backtesting Engines: The project now utilizes two distinct backtesters for different strategic approaches:

final_backtester.py: This is the engine for standard, end-of-period strategies. It operates on a fixed timeframe (e.g., daily, weekly) and makes all entry decisions based on completed candles for that timeframe.

final_backtester_immediate.py: This is the specialized engine for accelerated "fast entry" strategies. It uses a hybrid model where the setup is identified on a higher timeframe (weekly or monthly), but the entry trigger is scanned for and executed on a daily basis.

3. Core Trading Strategy Routines
The following describes the routines implemented in the two backtesting scripts.

A. Standard Strategy (final_backtester.py)
Timeframes: daily, 2day, weekly, monthly.

Entry Logic (End-of-Period): A trade is only considered on the closing day of a period (e.g., on Friday for a weekly timeframe). The entry decision is based on the pattern formed by the completed candles of that timeframe.

Exit Logic: Exits are checked daily against the stop-loss defined by the higher timeframe's entry conditions, providing a more realistic simulation of a live stop order.

B. Immediate Entry Strategy (final_backtester_immediate.py)
Timeframes: weekly-immediate, monthly-immediate.

Core Principle: To enter a trade as early as possible within a forming weekly or monthly candle, rather than waiting for it to close.

Entry Logic (Hybrid Timeframe):

HTF Setup: The script identifies a valid setup pattern on the completed higher-timeframe (HTF) charts (e.g., last week's candles). The setup requires one or more consecutive red HTF candles followed by a single bullish green HTF candle that closed above its 30-period EMA.

Fast Entry Trigger Price: The trigger price is calculated as the highest high of the completed red HTF candles only.

Daily Scan: The script's main loop runs daily. On each day of the current, forming HTF candle (e.g., on Monday, Tuesday, etc., of the current week), it checks the daily price action.

Execution: A trade is entered on the first day that the daily high crosses above the "Fast Entry Trigger Price," provided it has not already been breached on a previous day within the current HTF period.

Filter Application: All filters (Volume, Relative Strength) are applied using the daily data at the moment of the potential entry.

Risk Calculation: The initial stop-loss is calculated using a lookback on the daily chart, making the risk definition more granular.

Exit Logic: Exits are managed on a daily basis, identical to the standard backtester.

C. Universal Rules (Apply to Both Backtesters)
Entry Filters (Applied Sequentially):

Market Regime Filter: Prevents new long positions if the daily Nifty 200 index is below its 50-day EMA.

Relative Strength Filter: Requires a stock's 30-day return to be greater than the Nifty 200's 30-day return.

Volume Filter: Requires the entry day's volume to be at least 1.3x its 20-day average.

ATR Filter: (Implemented but disabled) Rejects trades if volatility is too high.

Position Sizing: Risks a fixed percentage of the current total portfolio equity on each trade. Equity is recalculated after exits but before new entries.

Trade Management (Two-Leg Exit Strategy):

Leg 1 (Partial Profit): 50% of the position is sold at a 1:1 risk/reward target. The stop-loss is then moved to breakeven.

Leg 2 (Trailing Stop): For any profitable trade, the stop-loss is trailed daily to the higher of breakeven or the low of the most recent green candle.

Final Exit: The entire remaining position is closed if the price hits the current stop-loss.

4. Data and File System Structure
Input Data: Raw data is in historical_data/. Processed data is in data/processed/, subdivided by timeframe (e.g., data/processed/daily/).

Filenames: Stock data is named {SYMBOL}_daily_with_indicators.csv. Index data is named NIFTY200_INDEX_daily_with_indicators.csv.

Output: Results are saved to backtest_logs/, including a .txt summary and a .csv detailed trade log with granular audit columns like portfolio_equity_on_entry and initial_shares.