Project Code Explained
This document provides a functional overview of the key Python scripts in the Nifty 200 Pullback Strategy project.

1. Data Pipeline Scripts
These scripts are responsible for acquiring and preparing all the data needed for backtesting.

fyers_equity_scraper.py & fyers_equity_scraper_15min.py
Purpose: These are the primary data acquisition tools. They connect to the Fyers API to download historical price data for all stocks listed in nifty200.csv.

Logic:

They handle the Fyers API authentication process.

They intelligently manage downloads, fetching only new data since the last run (incremental update) or performing a full historical download if no data exists.

To respect API limits, they download data in batches (6 months for daily, 90 days for 15-min).

The ..._15min.py version is specifically configured to fetch 15-minute candles and save them to a separate historical_data_15min/ directory.

fyers_nifty200_index_scraper_15min.py
Purpose: This is a specialized scraper dedicated to downloading 15-minute data for the Nifty 200 index, which is required for the hybrid model's real-time market filters.

Logic: It functions identically to the 15-minute equity scraper but is hardcoded to fetch data only for the NSE:NIFTY200-INDEX symbol.

calculate_indicators_clean.py
Purpose: This is the definitive data processing engine. It takes the raw daily data and creates the clean, indicator-rich files used by the backtesters for strategic analysis.

Logic:

It reads the raw daily CSVs from the historical_data/ folder.

It calculates all necessary indicators for the daily timeframe (e.g., ema_30, ema_50, volume_20_sma, return_30).

It then aggregates the daily data upwards to create the 2-Day, Weekly, and Monthly candles, ensuring all timeframes are perfectly synchronized. It correctly handles the Monday-Friday weekly aggregation.

It calculates indicators for these higher timeframes.

It saves all processed files with correct, descriptive names (e.g., ABB_weekly_with_indicators.csv) to the data/processed/ directory.

2. Backtesting Engine Scripts
These scripts run the trading simulations and generate performance reports.

final_backtester_benchmark_logger.py
Purpose: This script's sole function is to generate our "Golden Benchmark." It runs the original, flawed strategy with lookahead bias.

Logic:

It simulates an intraday entry on Day T.

The Flaw: It confirms the entry using filters (Volume, RS, Market Regime) based on the final, end-of-day data from Day T, giving it an impossible "crystal ball" advantage.

Enhanced Logging: It generates a comprehensive _all_setups_log.csv file, logging every single "perfect" setup it finds. For trades missed due to capital, it runs a hypothetical simulation to determine their outcome, providing a complete picture of the strategy's unconstrained potential.

final_backtester_v8_hybrid_optimized.py
Purpose: This is the current state-of-the-art, realistic backtesting engine, free of lookahead bias. It is the primary tool for current research.

Logic (Hybrid Model):

Pre-Market (T-1 Data): It first scans the processed daily charts to identify all stocks that had a valid price action setup on the previous day (T-1). This generates a "watchlist."

Intraday Scan (T Day Data): It then loops through the raw 15-minute candles of the current day (T) for stocks on the watchlist.

Execution: A trade is entered at the close of the first 15-minute candle where all real-time filters (Volume Velocity, Intraday Market Strength, etc.) are met, provided the price is within a predefined slippage limit.

Trade Management: All exits (stop-loss, profit target, trailing stop) are managed according to the universal rules defined in the Master Prompt.

Reporting: It generates a full suite of reports, including a summary, a detailed log of filled trades, and a comprehensive log of all setups identified (both filled and missed).
