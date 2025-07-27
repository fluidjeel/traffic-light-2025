Project Code Explained
Version: 3.0
Date: 2025-07-27

This document provides a detailed functional overview of each key Python script in the Nifty 200 Pullback Strategy project, including its importance, what it accomplishes, and its critical logic.

Table of Contents
Data Pipeline Scripts

fyers_scrapers

calculate_indicators_clean.py

Benchmark Generators (Lookahead Bias)

benchmark_generator_daily.py

benchmark_generator_htf.py

Realistic Simulators (Bias-Free)

simulator_daily_hybrid.py

simulator_htf_scout_sniper.py

Validation & Analysis Scripts

validate_daily_subset.py / validate_htf_subset.py

analyze_breakeven_impact.py

1. Data Pipeline Scripts
fyers_scrapers
Importance: These are the foundational scripts of the entire project. Without them, no research or backtesting is possible.

Accomplishment: They connect to the data vendor's API to download and save raw historical data. This includes daily OHLCV data for all stocks and indices, as well as 15-minute intraday data, which is critical for the bias-free simulators.

Critical Logic: The core logic involves iterating through the nifty200.csv list, formatting date ranges, and making paginated API calls, with robust error handling to manage failed downloads or missing data for specific symbols.

calculate_indicators_clean.py
Importance: This script is the central data processing engine. It ensures that all backtesting scripts work with a clean, consistent, and pre-calculated dataset, which significantly speeds up the backtesting process.

Accomplishment: It reads all the raw daily data, resamples it into higher timeframes (weekly, monthly), calculates all necessary technical indicators (EMAs, SMAs, RS, ATR) for every timeframe, and saves the enriched dataframes as CSV files in the data/processed directory.

Critical Logic: The most critical block is the resample() and agg() logic. It correctly aggregates daily data into weekly candles (e.g., using 'W-FRI') by taking the first open, max high, min low, last close, and sum of volume. This ensures the HTF data is accurate.

2. Benchmark Generators (Lookahead Bias)
benchmark_generator_daily.py
Importance: This script generates the "Golden Benchmark" for the daily strategy. It represents the absolute maximum theoretical performance and is the definitive superset against which the realistic simulator is validated.

Accomplishment: It runs the daily strategy with intentional lookahead bias. It uses the completed data from the trigger day (Day T) to make its entry and filtering decisions, resulting in a "perfect" trade log.

Critical Logic: The core of its lookahead bias is in the main loop. After identifying a setup on Day T-1, it uses the data from df.iloc[loc] (the candle for Day T) to check the Volume and RS filters. This use of future information is what defines it as a benchmark.

benchmark_generator_htf.py
Importance: This script generates the "Golden Benchmark" for the HTF strategy, serving as the superset for validating the "Scout and Sniper" model.

Accomplishment: It runs the HTF strategy with lookahead bias, finding weekly patterns and confirming them with daily breakouts, using future knowledge to apply its filters.

Critical Logic: Its lookahead bias is similar to the daily version but applied in a multi-timeframe context. It identifies a weekly setup and then checks for a daily breakout. On the day of the breakout, it uses the completed daily candle's data to apply the Volume and RS filters, which is not possible in real-time.

3. Realistic Simulators (Bias-Free)
simulator_daily_hybrid.py
Importance: This is the state-of-the-art, realistic backtesting engine for the daily strategy. Its results represent an achievable performance estimate.

Accomplishment: It identifies a setup on Day T-1 and attempts to execute it on Day T using 15-minute intraday data and real-time conviction filters, thereby eliminating lookahead bias.

Critical Logic: The most critical block is the watchlist generation logic. At the end of the loop for Day T, it identifies a setup pattern and applies all the benchmark's EOD filters (Volume, RS, Market Regime) using only the data available up to the close of Day T. If a stock passes, it is added to the watchlist for Day T+1. This is the correct, bias-free equivalent of the benchmark's filtering.

simulator_htf_scout_sniper.py
Importance: This is the most advanced and robust script in the project, designed to eliminate a subtle but powerful form of lookahead bias present in HTF strategies.

Accomplishment: It uses the "Scout and Sniper" architecture to separate the discovery of a confirmed breakout from the decision to enter a trade.

Critical Logic:

The Scout (EOD): At the end of Day T, the Scout's logic is paramount. It scans for weekly patterns and then checks the completed daily candle of Day T to see if a breakout has already occurred and if all EOD filters were met. This generates a high-probability target list for the next day.

The Sniper (Intraday): On Day T+1, the Sniper's logic is purely about conviction. It does not check for a price breakout. It only monitors the target list and waits for a 15-minute candle to satisfy real-time filters like volume_velocity_filter, confirming that the breakout from the previous day has momentum.

4. Validation & Analysis Scripts
validate_daily_subset.py / validate_htf_subset.py
Importance: These are essential diagnostic tools that ensure the logical integrity of the entire backtesting framework.

Accomplishment: They compare the all_setups_log.csv files generated by a benchmark and its corresponding simulator. They mathematically prove that the simulator is a true subset of the benchmark, confirming that it is not inventing trades and that its filtering logic is working as expected.

Critical Logic: The core of the script is the use of Python's set operations. It creates a set of setup_ids from both logs and performs an issubset() check. This is a simple but powerful way to validate the relationship between the two scripts.

analyze_breakeven_impact.py
Importance: This is a modular analysis script that allows for the isolated study of a single strategy feature.

Accomplishment: It consumes a trades_detail.csv file from a simulator run and calculates the specific P&L contribution of the use_aggressive_breakeven rule. This allows for data-driven decisions on whether a feature is beneficial or detrimental to the strategy's performance.

Critical Logic: It iterates through the trade log and recalculates what the final P&L of each trade would have been if the aggressive breakeven rule had not been applied, then compares this hypothetical result to the actual result.