Project Runbook & Operational Guide (Version 5.0)
This document provides the Standard Operating Procedure (SOP) for running the Traffic Light 2025 project's modern research pipeline and a detailed guide to the day-wise activities for live or simulated execution using the latest tfl simulators.

Part 1: Research & Backtesting Workflow
This section outlines the standard process for conducting research and running backtests from start to finish. Following these steps in order is critical for maintaining data integrity and producing valid, bias-free results.

Step 1: Data Acquisition (Universal Pipeline)
Ensure all historical data is up-to-date using the new universal scraper. This single script handles all data downloading tasks.

To perform a full, fresh download or an intelligent incremental update of all data (Daily & 15-min):

python universal_fyers_scraper.py

To force a complete re-download of all data from the beginning:

python universal_fyers_scraper.py --force

To update only a specific interval (e.g., daily data):

python universal_fyers_scraper.py --interval daily

Step 2: Data Processing (Universal Pipeline)
Process the raw data downloaded in Step 1 to create the necessary indicator files for all required timeframes. This step is mandatory after any data update.

Run the Universal Indicator Calculator:

python universal_calculate_indicators.py

Step 3: Run Realistic Simulators
Execute the bias-free backtests to get realistic performance estimates. This is the core of the research process.

Configure the Script: Open the relevant simulator (daily_tfl_simulator.py, weekly_tfl_simulator.py, or monthly_tfl_simulator.py) and adjust the primary parameters in the config dictionary.

Select the Data Pipeline: Inside the config, locate the data_pipeline_config section and set 'use_universal_pipeline': True to use the new data, or False to use legacy data folders.

Execute the Backtest:

For Daily Strategy: python daily_tfl_simulator.py

For Weekly HTF Strategy: python weekly_tfl_simulator.py

For Monthly Strategy: python monthly_tfl_simulator.py

Step 4: Perform Advanced Stop-Loss Analysis
After generating one or more backtest logs, use the MAE Analyzer to perform a data-driven "what-if" analysis on your stop-loss strategy.

Run the MAE Analyzer on a specific strategy's logs:

python mae_analyzer_percent.py --strategy <strategy_name>

(Replace <strategy_name> with the folder name, e.g., monthly_tfl_simulator)

Step 5: Analyze All Results
Navigate to the newly created subdirectory within backtest_logs (e.g., backtest_logs/monthly_tfl_simulator/). A thorough analysis should proceed as follows:

Start with the Summary (_summary.txt): Get a high-level overview of the performance and confirm the input parameters.

Review the MAE Analysis Report: Use the output from the mae_analyzer_percent.py script to determine an optimal stop-loss range.

Deep Dive into Executed Trades (_trade_details.csv): Analyze the P&L distribution, holding periods, and reasons for exits.

Analyze Opportunity Cost (_missed_trades.csv): Analyze how many high-potential trades were missed due to capital or risk limits.

Fine-Tune the Conviction Engine (_filtered.csv): Analyze which filters are rejecting the most trades to determine if they are too strict.

Part 2: Day-wise Execution Guide for Simulators
This section details the specific, timed activities required to run the realistic simulators, replicating a live trading environment.

A. Daily Strategy Execution (daily_tfl_simulator.py)
This strategy identifies a setup on Day T-1 and attempts to execute it on Day T.

Post-Market / End-of-Day (EOD) on Day T-1: The script scans the completed daily data, identifies patterns, applies EOD quality filters, and generates a watchlist for the next trading day.

Intraday during Day T: The script monitors 15-minute data for stocks on the watchlist. When a breakout occurs, it immediately checks the real-time Conviction Engine before simulating an entry.

B. HTF (Weekly) Strategy Execution (weekly_tfl_simulator.py)
This strategy uses the "Scout and Sniper" model to separate weekly setup discovery from intraday entry execution.

Post-Market / End-of-Day (EOD) on Friday: The "Scout" scans the completed weekly charts to identify all valid pullback patterns and generates a "Target List" for the entire following week.

Intraday during the Following Week (Monday - Friday): The "Sniper" monitors 15-minute data for stocks on the Target List and validates any breakout using the full Conviction Engine.

C. Monthly Strategy Execution (monthly_tfl_simulator.py)
This strategy adapts the Scout/Sniper model for a much longer timeframe.

Post-Market / End-of-Day (EOD) on the Last Trading Day of the Month: The "Scout" scans the completed monthly charts to identify valid pullback patterns and generates a "Target List" valid for the entire next month.

Intraday during the Following Month: The "Sniper" monitors 15-minute data for stocks on the month's Target List, operating within an Adaptive Execution Window and validating breakouts against the Conviction Engine.