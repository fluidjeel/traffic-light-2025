Project Runbook & Operational Guide
This document provides the Standard Operating Procedure (SOP) for running the Nifty 200 Pullback Strategy's research pipeline. Following these steps in order is critical for generating accurate and comparable backtest results.

Step 1: Data Acquisition
Objective: To ensure you have the latest raw daily and 15-minute price data for all Nifty 200 stocks and the required indices (Nifty 200, India VIX).

Procedure:

Authenticate with Fyers: Ensure your fyers_access_token.txt is valid. If in doubt, delete it and re-authenticate by running any scraper.

Run the Daily Equity Scraper:

python fyers_equity_scraper.py

Run the Daily Index Scraper (New Step):

python fyers_index_scraper.py

Run the 15-Minute Equity Scraper:

python fyers_equity_scraper_15min.py

Step 2: Data Processing
Objective: To process the raw daily data and generate the clean, indicator-rich files needed by the backtesters.

Procedure:

Run the Indicator Calculator:

python calculate_indicators_clean.py

This script will now automatically find and process the INDIAVIX_daily.csv file created in the previous step.

Step 3: Generate the "Golden Benchmark"
Objective: To run the original, flawed backtester to generate the "perfect" trade log that serves as our performance benchmark.

Procedure:

Execute the Backtest:

python final_backtester_benchmark_logger.py

Step 4: Run the Realistic Hybrid Backtest
Objective: To run the current state-of-the-art, realistic backtester and analyze the impact of its various features.

Procedure:

Configure the Script: Open final_backtester_v8_hybrid_optimized.py. The config dictionary at the top of the file now contains several experimental toggles. Adjust these boolean flags (True/False) to test different combinations of rules:

'use_dynamic_slippage': Enables the realistic slippage model based on liquidity and VIX.

'cancel_on_gap_up': Prevents entry if a stock opens above its trigger price.

'prevent_entry_below_trigger': Prevents entry if the price falls below the trigger after confirmation filters are met.

'use_aggressive_breakeven': Moves the stop to just above breakeven on profitable EOD positions.

'use_partial_profit_leg': Enables the 1:1 risk/reward partial profit target. Set to False to let the entire position run.

Execute the Backtest:

python final_backtester_v8_hybrid_optimized.py

Review and Compare: The script will generate its own set of logs. Analyze the summary report and the detailed trade logs to understand the impact of the enabled/disabled features on performance.

Step 5: Analyze Feature Impact (Optional)
Objective: To get a detailed, quantitative analysis of the impact of a specific experimental feature.

Procedure:

Run the Breakeven Impact Analyzer: To specifically analyze the net P&L impact of the 'use_aggressive_breakeven' feature, run the following command after completing Step 4 with the feature enabled.

python analyze_breakeven_impact.py

Review Analysis Report: The script will print a summary in the terminal, providing a data-driven conclusion on whether the feature is beneficial or detrimental to the strategy's profitability.Master Project Context Prompt: Nifty 200 Pullback Strategy
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. The goal is for you, the LLM, to fully understand the project's architecture, data flow, and the precise logic of the two distinct trading strategies to assist with future modifications, bug fixes, or the addition of new features. Do not assume any prior knowledge; this document is the single source of truth.

1. High-Level Project Goal & Status
1.1. Goal:
The project's objective is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge.

1.2. Current Status:
The project is in an advanced research and development phase. The backtesting framework (final_backtester_v8_hybrid_optimized.py) is stable and has been enhanced with a suite of toggleable, experimental features for entry filtering and trade management. The current focus is on systematically testing these new features to find the optimal configuration that maximizes risk-adjusted returns.

2. System Architecture & Script Workflow
The project is a modular data pipeline. Each script performs a distinct, specialized task.

2.1. Script Pipeline:
Data Acquisition:

fyers_equity_scraper.py: Downloads daily equity data.

fyers_index_scraper.py: Downloads daily index data (Nifty 200, India VIX).

fyers_equity_scraper_15min.py: Downloads 15-minute equity data.

Indicator Calculation (calculate_indicators_clean.py): Takes raw daily data, calculates all necessary indicators, and saves the processed files.

Strategy Simulation (Two Engines):

final_backtester_benchmark_logger.py: Runs a flawed strategy with lookahead bias to generate a "Golden Benchmark" of ideal trades.

final_backtester_v8_hybrid_optimized.py: The primary, realistic backtester with a full suite of experimental features.

Post-Hoc Analysis (analyze_breakeven_impact.py): A dedicated script to analyze the net P&L impact of the "Aggressive Breakeven" feature.

3. Core Trading Strategy: Hybrid Intraday Model
This section defines the logic of the primary, realistic backtesting engine.

3.1. Setup Identification (Pre-Market)
Pattern Recognition: At the start of Day T, the system scans the daily charts from the previous day (T-1) to find a specific pattern: one or more red candles followed by a single green candle that closes above the 30-period EMA.

Trigger Price: A trigger_price is calculated as the highest high of this entire red + green candle pattern.

Gap-Up Filter (Toggleable): If cancel_on_gap_up is enabled, the system checks the opening price of the first 15-minute candle. If it's already above the trigger_price, the setup is cancelled for the day.

3.2. Entry Execution (Intraday)
The system monitors the 15-minute chart for all valid setups. A trade is entered at the close of the first 15-minute candle that meets all of the following conditions:

Price Trigger: The candle's high must have crossed above the trigger_price.

Volume Velocity Filter: The cumulative intraday volume must exceed a percentage of the stock's 20-day average daily volume.

Market Strength Filter: The Nifty 200 index must be trading above its opening price for the day.

Failed Breakout Filter (Toggleable): If prevent_entry_below_trigger is enabled, the 15-minute candle's closing price must still be above the trigger_price.

Fill Price Simulation: The final entry price is adjusted by a Dynamic Slippage Model (if enabled), which calculates a realistic cost based on the stock's liquidity (average volume) and market volatility (VIX).

3.3. Position & Trade Management
Position Sizing: Risks a fixed percentage of the total portfolio equity on each trade.

Initial Stop-Loss: The lowest low of the 5 daily candles immediately preceding the entry day.

Trade Management (Two-Leg Exit Strategy):

Leg 1 (Partial Profit - Toggleable): If use_partial_profit_leg is enabled, 50% of the position is sold at a 1:1 risk/reward target. The stop-loss for the remaining shares is then moved to the entry price.

Leg 2 (Trailing Stop): The stop-loss is managed daily using two primary rules:

Standard Trail: The stop is trailed to the low of the most recent green daily candle.

Aggressive Breakeven (Toggleable): If use_aggressive_breakeven is enabled, the stop is moved to slightly above the entry price as soon as a trade is profitable at the end of the day. This is logged with a unique exit type (Aggressive Breakeven Stop) for analysis.

The final stop-loss for any given day is the highest value calculated by any of the applicable trailing stop rules.