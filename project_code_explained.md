Project Code Explained
This document provides a functional overview of the key Python scripts in the Nifty 200 Pullback Strategy project.

1. Data Pipeline Scripts
These scripts are responsible for acquiring and preparing all the data needed for backtesting.

fyers_equity_scraper.py & fyers_equity_scraper_15min.py

Purpose: Primary data acquisition tools for all stocks listed in nifty200.csv. They intelligently manage incremental updates to respect API limits.

fyers_index_scraper.py (New)

Purpose: A specialized scraper dedicated to downloading daily data for key indices, specifically NIFTY200_INDEX and INDIAVIX. The VIX data is essential for the dynamic slippage model.

calculate_indicators_clean.py

Purpose: The definitive data processing engine. It takes raw daily data for all equities and indices, calculates all necessary indicators (EMAs, SMAs, etc.), and saves the final, clean files to the data/processed/ directory.

2. Backtesting Engine Scripts
These scripts run the trading simulations and generate performance reports.

final_backtester_benchmark_logger.py

Purpose: This script's sole function is to generate our "Golden Benchmark." It runs the original, flawed strategy with lookahead bias, using end-of-day data to confirm intraday entries.

Enhanced Logging: It generates a comprehensive _all_setups_log.csv file, logging every single "perfect" setup it finds.

final_backtester_v8_hybrid_optimized.py

Purpose: This is the current state-of-the-art, realistic backtesting engine, free of lookahead bias. It is the primary tool for current research and has been significantly enhanced with multiple experimental features.

Logic (Hybrid Model):

Pre-Market (T-1 Data): It first scans the processed daily charts to identify all stocks that had a valid price action setup on the previous day (T-1). This generates a "watchlist."

Pre-Market Filter (Toggleable): If cancel_on_gap_up is enabled, it checks the opening price of the first 15-min candle. If this price is above the setup's trigger price, the trade is cancelled before the main session begins.

Intraday Scan (Day T Data): It loops through the 15-minute candles of the current day (T) for stocks on the watchlist.

Execution & Filtering: A trade is entered at the close of the first 15-minute candle where all real-time filters are met. The key filters are:

Volume Velocity: Checks if cumulative intraday volume has met a certain threshold of the 20-day average.

Market Strength: Checks if the Nifty 200 index is positive for the day.

Failed Breakout (Toggleable): If prevent_entry_below_trigger is enabled, it ensures the price has not fallen back below the trigger price at the moment of execution.

Fill Price Simulation: The entry price is adjusted using a Dynamic Slippage Model (if enabled) that calculates a realistic transaction cost based on the stock's liquidity and market volatility (VIX).

Trade Management: All exits are managed according to a set of rules, now with experimental toggles:

Partial Profit Target (Toggleable): If use_partial_profit_leg is enabled, 50% of the position is sold at a 1:1 risk/reward target.

Aggressive Breakeven (Toggleable): If use_aggressive_breakeven is enabled, the stop-loss is moved to slightly above the entry price as soon as a trade is profitable at the end of the day. This is logged with a unique exit type for later analysis.

Standard Trailing Stop: The stop-loss is trailed daily using the low of the most recent green candle.

Reporting: It generates a full suite of reports, including a summary, a detailed log of filled trades (with specific exit types), and a comprehensive log of all setups identified (including filled, missed, and cancelled).