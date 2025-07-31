Project Runbook & Operational Guide (Version 4.4)
This document provides the Standard Operating Procedure (SOP) for running the Nifty 200 Pullback Strategy's research pipeline and a detailed guide to the day-wise activities for live or simulated execution using the latest simulators.

Part 1: Research & Backtesting Workflow
This section outlines the standard process for conducting research and running backtests from start to finish. Following these steps in order is critical for maintaining data integrity and producing valid results.

Step 1: Data Acquisition
Ensure all historical data is up-to-date. This is a four-step process that must be completed before any other script is run.

Run the Daily Equity Scraper: python fyers_equity_scraper.py

Run the Daily Index Scraper: python fyers_index_scraper.py

Run the 15-Minute Equity Scraper: python fyers_equity_scraper_15min.py

Run the 15-Minute Nifty 200 Index Scraper: python fyers_nifty200_index_scraper_15min.py

Step 2: Data Processing
Process the raw data downloaded in Step 1 to create the necessary indicator files for all required timeframes. This step is mandatory after any data update.

Run the Indicator Calculator: python calculate_indicators_clean.py

Step 3: Generate "Golden Benchmarks" (Optional but Recommended)
Generate the theoretical performance ceilings for comparison. This helps validate that the realistic simulators are performing within expected logical bounds.

Generate Daily Benchmark: python benchmark_generator_daily.py

Generate HTF Benchmark: python benchmark_generator_htf.py

Step 4: Run Realistic Simulators
Execute the bias-free backtests to get realistic performance estimates. This is the core of the research process.

Configure the Script: Open the relevant simulator (simulator_daily_hybrid.py, htf_simulator_advanced.py, or simulator_monthly_advanced.py) and adjust the primary parameters in the config dictionary.

Configure Logging Options: Within the config dictionary, locate the log_options section. Adjust the boolean flags (True/False) for each report type to control which detailed log files are generated.

Execute the Backtest:

For Daily Strategy: python simulator_daily_hybrid.py

For Weekly HTF Strategy: python htf_simulator_advanced.py

For Monthly Strategy: python simulator_monthly_advanced.py

Step 5: Perform Advanced Stop-Loss Analysis (New)
After generating one or more backtest logs, use the MAE Analyzer to perform a data-driven "what-if" analysis on your stop-loss strategy.

Run the MAE Analyzer in Default Mode:

python mae_analyzer.py

This will automatically find and analyze all _trade_details.csv files in the backtest_logs/simulator_monthly_advanced/ directory.

Run the MAE Analyzer on a Specific File:

python mae_analyzer.py --file <path_to_log_file.csv>

Step 6: Analyze All Results (Enhanced Workflow)
Navigate to the newly created subdirectory within backtest_logs (e.g., backtest_logs/simulator_monthly_advanced/). A thorough analysis should proceed as follows:

Start with the Summary (_summary.txt): Get a high-level overview of the performance and confirm the input parameters.

Review the MAE Analysis Report: Use the output from the mae_analyzer.py script to determine an optimal stop-loss range. The recommendation will highlight the "sweet spot" that best balances cutting losses and preserving winning trades.

Deep Dive into Executed Trades (_trade_details.csv): Analyze the P&L distribution, holding periods, and reasons for exits. Use the new mae_percent column to see how much drawdown your winning trades typically experienced.

Analyze Opportunity Cost (_missed_trades.csv): Analyze how many high-potential trades were missed due to capital or risk limits.

Fine-Tune the Conviction Engine (_filtered.csv): Analyze which filters are rejecting the most trades to determine if they are too strict.

Step 7: Perform Validation
Verify the logical integrity of the simulators against their benchmarks. This step should be performed whenever a significant change is made to the core strategy logic.

Run the Daily Validation:
python validate_daily_subset.py <path_to_benchmark_daily_log.csv> <path_to_simulator_daily_log.csv>

Run the HTF Validation:
python validate_htf_subset.py <path_to_benchmark_htf_log.csv> <path_to_simulator_htf_log.csv>

Part 2: Day-wise Execution Guide for Simulators
This section details the specific, timed activities required to run the realistic simulators, replicating a live trading environment.

A. Daily Strategy Execution (simulator_daily_hybrid.py)
This strategy identifies a setup on Day T-1 and attempts to execute it on Day T.

Post-Market / End-of-Day (EOD) on Day T-1:

Run Watchlist Generation: The script scans the completed daily data, identifies patterns, applies EOD quality filters, and generates a watchlist for the next trading day.

Intraday during Day T:

Monitor for Entries: The script monitors 15-minute data for stocks on the watchlist. When a breakout occurs, it immediately checks the real-time Advanced Conviction Engine before simulating an entry.

B. HTF (Weekly) Strategy Execution (htf_simulator_advanced.py)
This strategy uses the "Scout and Sniper" model to separate weekly setup discovery from intraday entry execution.

Post-Market / End-of-Day (EOD) on Friday:

Run the "Scout": The Scout scans the completed weekly charts to identify all valid pullback patterns and generates a "Target List" for the entire following week.

Intraday during the Following Week (Monday - Friday):

"Sniper" Monitors for Conviction: The Sniper monitors 15-minute data for stocks on the Target List. When a breakout occurs, it instantly validates the move using the full Advanced Conviction Engine.

C. Monthly Strategy Execution (simulator_monthly_advanced.py)
This strategy adapts the Scout/Sniper model for a much longer timeframe.

Post-Market / End-of-Day (EOD) on the Last Trading Day of the Month:

Run the "Scout": The Scout scans the completed monthly charts to identify valid pullback patterns. It applies any configured EOD quality filters (like trend or volume) and generates a "Target List" valid for the entire next month.

Intraday during the Following Month:

"Sniper" Monitors within an Adaptive Window: The Sniper monitors 15-minute data for stocks on the month's Target List. It operates within an Adaptive Execution Window, typically skipping the first few days of the month and then actively monitoring for a set period (e.g., 15 trading days, extended during high volatility). When a stock breaks its trigger price, the Sniper validates the move against the Conviction Engine before execution.