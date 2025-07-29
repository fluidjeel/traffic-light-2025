Project Runbook & Operational Guide (Version 4.0)
This document provides the Standard Operating Procedure (SOP) for running the Nifty 200 Pullback Strategy's research pipeline and a detailed guide to the day-wise activities for live or simulated execution using the latest simulators.

Part 1: Research & Backtesting Workflow
This section outlines the standard process for conducting research and running backtests.

Step 1: Data Acquisition
Ensure all historical data is up-to-date. This is a four-step process.

Run the Daily Equity Scraper: python fyers_equity_scraper.py

Run the Daily Index Scraper: python fyers_index_scraper.py

Run the 15-Minute Equity Scraper: python fyers_equity_scraper_15min.py

Run the 15-Minute Nifty 200 Index Scraper: python fyers_nifty200_index_scraper_15min.py

Step 2: Data Processing
Process the raw data to create the necessary indicator files for all timeframes.

Run the Indicator Calculator: python calculate_indicators_clean.py

Step 3: Generate "Golden Benchmarks" (Optional but Recommended)
Generate the theoretical performance ceilings for comparison.

Generate Daily Benchmark: python benchmark_generator_daily.py

Generate HTF Benchmark: python benchmark_generator_htf.py

Step 4: Run Realistic Simulators
Execute the bias-free backtests to get realistic performance estimates.

Configure the Script: Open the relevant simulator (simulator_daily_hybrid-v2.py or htf_simulator_advanced.py) and adjust the parameters in the config dictionary.

Execute the Backtest:

For Daily Strategy: python simulator_daily_hybrid-v2.py

For HTF Strategy: python htf_simulator_advanced.py

Step 5: Perform Validation
Verify the logical integrity of the simulators against their benchmarks.

Run the Daily Validation:
python validate_daily_subset.py <path_to_benchmark_daily_log.csv> <path_to_simulator_daily_log.csv>

Run the HTF Validation:
python validate_htf_subset.py <path_to_benchmark_htf_log.csv> <path_to_simulator_htf_log.csv>

Part 2: Day-wise Execution Guide for Simulators
This section details the specific, timed activities required to run the realistic simulators, replicating a live trading environment.

A. Daily Strategy Execution (simulator_daily_hybrid-v2.py)
This strategy identifies a setup on Day T-1 and attempts to execute it on Day T.

Post-Market / End-of-Day (EOD) on Day T-1 (e.g., Monday, after 3:30 PM)

Run Watchlist Generation: Execute the watchlist generation portion of the script.

What it Does: The script scans the completed daily data from Day T-1. It identifies all stocks that meet the core pattern and pass all the end-of-day quality filters.

Output: A watchlist of high-probability setups is generated for the next trading day, Day T.

Intraday during Day T (e.g., Tuesday, 9:15 AM - 3:30 PM)

Monitor for Entries: The script continuously monitors the 15-minute data for stocks on the watchlist.

What it Does: When an intraday candle's high crosses the trigger price, it immediately checks the real-time Advanced Conviction Engine. If all checks pass, an entry order is simulated.

Monitor Open Positions: The script also monitors 15-minute data for all open positions to manage exits.

B. HTF Strategy Execution (htf_simulator_advanced.py)
This strategy uses the "Scout and Sniper" model to separate weekly setup discovery from intraday entry execution.

Post-Market / End-of-Day (EOD) on Friday

Run the "Scout": Execute the scout_for_setups portion of the htf_simulator_advanced.py script.

What it Does: The Scout scans all stocks against the weekly data that has just completed. It identifies all valid weekly pullback patterns that meet the definitive strategy criteria.

Output: A high-probability "Target List" is generated for the entire following week (Monday-Friday). This list includes the symbol, its trigger price, and its target daily volume.

Intraday during the Following Week (Monday - Friday, 9:15 AM - 3:30 PM)

"Sniper" Monitors for Conviction: The Sniper monitors the 15-minute data for stocks on the active Target List.

What it Does: It is not waiting for a simple price breakout. When a stock on the list breaks its trigger price, the Sniper instantly validates the move using the full Advanced Conviction Engine (Volume Projection, VIX-Adaptive Strength, Intraday RS).

Execute & Manage: If and only if all conviction checks pass, an entry is simulated. Open positions are also managed intraday for exits based on the dynamic profit target and trailing stop-loss logic.