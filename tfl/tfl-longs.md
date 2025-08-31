Documentation: TrafficLight-Manny (Longs) Simulator
1. System Overview
This document provides a comprehensive explanation of the longs_simulator.py script and its role within the "TrafficLight-Manny" quantitative trading system.

The simulator is a parallelized, single-symbol backtester. It is designed to test the performance of the "TrafficLight-Manny (Longs)" strategy across a large universe of instruments by running an independent backtest for each symbol and then aggregating the results. It is not a portfolio-level simulator (i.e., it does not manage a single pool of capital across simultaneous trades), but rather a powerful tool for validating the strategy's rules on a large scale.

The system is built on a robust data pipeline that pre-calculates all necessary indicators to ensure high-speed, lookahead-bias-free execution.

2. Folder Structure & Relevant Files
For the simulator to function correctly, the project must be organized with the following folder structure:

D:\algo-2025\
│
├── tfl\
│   ├── longs_simulator.py        # <-- The simulator script
│   └── ... (other strategy scripts)
│
├── data\
│   └── universal_processed\
│       ├── 15min\
│       │   ├── [SYMBOL]_15min_with_indicators.parquet
│       │   └── ...
│       └── daily\
│           ├── [SYMBOL]_daily_with_indicators.parquet
│           └── ...
│
├── backtest_logs\
│   └── TrafficLight-Manny-LONGS_ONLY\
│       └── [TIMESTAMP]\
│           ├── summary.txt
│           └── trade_log.csv
│
└── nifty200_fno.csv               # List of symbols to trade

Relevant Files
longs_simulator.py: The core backtesting engine. It reads the pre-processed data, simulates trades based on the strategy rules, and generates detailed performance logs.

[SYMBOL]_15min_with_indicators.parquet: The primary data files containing 15-minute candle data and all pre-calculated technical indicators (EMAs, MVWAP, ATR, etc.).

[SYMBOL]_daily_with_indicators.parquet: Data files containing daily candle data and daily indicators (like the daily RSI).

nifty200_fno.csv: A simple CSV file containing a list of all stock symbols to be included in the backtest run.

summary.txt: The output file that contains the final performance metrics and the configuration settings used for the run.

trade_log.csv: The detailed output file containing a row for every single trade executed during the simulation.

3. Strategy Rules (TrafficLight-Manny Longs)
The simulator executes trades based on a precise set of rules, which are the symmetrical opposite of the shorts strategy.

Entry Rules
A long trade is initiated on a 15-minute candle if ALL of the following conditions are met:

Price Action Pattern: A sequence of 1 to 9 consecutive red candles is immediately followed by a green "signal" candle.

Entry Trigger: The high of the candle after the signal candle breaks above the high of the full red-and-green candle pattern.

Daily Momentum Filter: The daily RSI (14-period) of the instrument must be above 75, indicating it is in a state of strong bullish momentum.

Intraday Trend Filter (Adaptive):

For stocks, the price must be trading above its 50-period Moving Volume Weighted Average Price (MVWAP).

For indices, the price must be trading above its 50-period Exponential Moving Average (EMA).

Trade Management & Exits
Initial Stop Loss: Placed at the lowest low of the identified pattern candles.

Initial Take Profit: A distant target is set at 10R (10 times the initial risk).

Breakeven Stop: Once the trade reaches a profit of 1.0R, the stop-loss is moved to lock in a small profit of 0.1R.

Multi-Stage ATR Trailing Stop:

A standard trailing stop is active from the start, using a 4.0x multiplier on the 14-period ATR.

If the trade becomes highly profitable and reaches 5.0R, the trailing stop becomes more aggressive, tightening to a 1.0x multiplier on the ATR.

Trading Mode: The simulator can be run in two modes:

INTRADAY: All open positions are automatically closed at the 15:15 EOD candle.

POSITIONAL: Trades can be held overnight. The simulator includes logic to handle overnight gaps correctly.

4. Code Explanation
The longs_simulator.py script is divided into several key functions:

main()
This is the orchestrator function that controls the entire backtesting process.

It sets up the logging directory for the current run.

It reads the nifty200_fno.csv to get the list of symbols to backtest.

It pre-loads all the necessary daily data into a daily_data_map for high-speed access.

It uses Python's multiprocessing.Pool to distribute the backtesting of each symbol across all available CPU cores, dramatically speeding up the process.

After all individual backtests are complete, it aggregates the trade logs from all symbols into a single trade_log.csv and calculates the overall performance metrics.

process_symbol_backtest(symbol, symbol_daily_data)
This function acts as a wrapper for each parallel process. Its job is to:

Load the 15-minute data for the specific symbol it has been assigned.

Filter the data for the correct date range.

Call the run_backtest function to perform the actual simulation.

Return the results as a pandas DataFrame.

run_backtest(df, symbol, symbol_daily_data)
This is the core of the simulator where the strategy logic is executed.

It starts by merging the daily data (like the daily RSI) onto the 15-minute DataFrame, ensuring no lookahead bias.

It then uses vectorized pandas and NumPy operations to efficiently identify every potential trade setup (the red-then-green candle pattern) in the entire dataset at once.

It then enters a for loop that iterates through the data candle by candle.

If a trade is active, it checks for all exit conditions (SL, TP, trailing stops, EOD exit) and manages the trade according to the rules.

If no trade is active, it checks if a valid entry trigger has occurred. If it has, it applies all the filters (RSI, MVWAP/EMA). If all conditions are met, it initiates a new trade.

It returns a list of all completed trades for that symbol.

log_configs() & calculate_and_log_metrics()
These are helper functions responsible for writing the configuration settings and the final performance summary to the summary.txt log file.

5. Next Step - Realistic Simulator Development
This section provides instructions for using an LLM to transition this single-symbol backtester into a full portfolio-level simulator.

Step 1: Create a Master Data File with Plotted Signals
The first step is to create a single, unified data file that contains the 15-minute data for all symbols, with the long entry signals already pre-calculated.

Prompt to LLM: "Please create a new Python script named create_long_signals_parquet.py. This script should read the individual 15-minute Parquet files for all symbols in my universe, apply the exact long signal logic from longs_simulator.py to pre-calculate and flag all potential entry signals, and save the combined data for all symbols into a single, symbol-partitioned Parquet file named tfl_longs_data_with_signals.parquet."

Step 2: Explain Existing Regime Data
The market regime data is universal and does not need to be recreated.

Explanation for LLM: "The system already has a market_regime_data.parquet file. This file contains daily market breadth, volatility (VIX), and NIFTY 50 trend data. This single file should be used by the new portfolio simulator for its regime filters."

Step 3: Create the Portfolio-Level Simulator
The final step is to create the new portfolio simulator that operates on the master data file.

Prompt to LLM: "Please create a new portfolio-level simulator named long_portfolio_simulator_realistic.py. This script should not run backtests one by one. Instead, it should:

Load the master tfl_longs_data_with_signals.parquet and the market_regime_data.parquet files.

Iterate through every unique 15-minute timestamp in the master data file.

At each timestamp, it should manage a single portfolio with a unified capital base. It should check for exits on all open positions across all symbols, and then scan for new entry signals across all symbols.

It must enforce a strict portfolio-level limit on the maximum number of concurrent open positions (e.g., 15).

If more signals appear than available slots, it must prioritize them by the highest daily RSI."