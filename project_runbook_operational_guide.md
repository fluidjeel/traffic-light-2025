Project Runbook & Operational Guide (Version 3.1)
This document provides the Standard Operating Procedure (SOP) for running the Nifty 200 Pullback Strategy's research pipeline and a detailed guide to the day-wise activities for live or simulated execution.

Part 1: Research & Backtesting Workflow
This section outlines the standard process for conducting research and running backtests.

Step 1: Data Acquisition
Ensure all historical data is up-to-date. This is a four-step process.

Run the Daily Equity Scraper: python fyers_equity_scraper.py

Run the Daily Index Scraper: python fyers_index_scraper.py

Run the 15-Minute Equity Scraper: python fyers_equity_scraper_15min.py

Run the 15-Minute Nifty 200 Index Scraper: python fyers_nifty200_index_scraper_15min.py

Step 2: Data Processing
Process the raw data to create the necessary indicator files.

Run the Indicator Calculator: python calculate_indicators_clean.py

Step 3: Generate "Golden Benchmarks" (Optional but Recommended)
Generate the theoretical performance ceilings for comparison.

Generate Daily Benchmark: python benchmark_generator_daily.py

Generate HTF Benchmark: python benchmark_generator_htf.py

Step 4: Run Realistic Simulators
Execute the bias-free backtests to get realistic performance estimates.

Configure the Script: Open the relevant simulator (simulator_daily_hybrid.py or simulator_htf_scout_sniper.py) and adjust the parameters in the config dictionary.

Execute the Backtest:

For Daily Strategy: python simulator_daily_hybrid.py

For HTF Strategy: python simulator_htf_scout_sniper.py

Step 5: Perform Validation
Verify the logical integrity of the simulators against their benchmarks.

Run the Daily Validation:
python validate_daily_subset.py <path_to_benchmark_daily_log.csv> <path_to_simulator_daily_log.csv>

Run the HTF Validation:
python validate_htf_subset.py <path_to_benchmark_htf_log.csv> <path_to_simulator_htf_log.csv>

Part 2: Day-wise Execution Guide for Simulators
This section details the specific, timed activities required to run the realistic simulators, replicating a live trading environment.

A. Daily Strategy Execution (simulator_daily_hybrid.py)

This strategy identifies a setup on Day T-1 and attempts to execute it on Day T.

Post-Market / End-of-Day (EOD) on Day T-1 (e.g., Monday, after 3:30 PM)

Run Watchlist Generation: Execute the watchlist generation portion of the simulator_daily_hybrid.py script.

What it Does: The script scans the completed daily data from Day T-1. It identifies all stocks that meet the core pattern (green candle preceded by reds) and pass all the end-of-day quality filters (EMA, Volume, RS, etc.).

Output: A watchlist of high-probability setups is generated for the next trading day, Day T.

Pre-Market on Day T (e.g., Tuesday, before 9:15 AM)

Run Gap-Up Filter: Execute the pre-market check within the simulator.

What it Does: It checks the opening price for all stocks on the watchlist. Any stock that is set to open significantly above its trigger price is removed from the active watchlist for the day to avoid chasing an excessive gap.

Intraday during Day T (e.g., Tuesday, 9:15 AM - 3:30 PM)

Monitor for Entries: The script continuously monitors the 15-minute data for stocks remaining on the watchlist.

What it Does: When an intraday candle closes above the trigger price, it immediately checks the real-time Advanced Conviction Engine (VIX-adaptive market strength, volume projection, etc.). If all checks pass, an entry order is simulated.

Monitor Open Positions: The script also monitors 15-minute data for all open positions to manage exits based on the partial profit target or the current stop-loss level.

B. HTF Strategy Execution (simulator_htf_scout_sniper.py)

This strategy uses the "Scout and Sniper" model to separate breakout confirmation from entry execution.

Post-Market / End-of-Day (EOD) on Day T (e.g., Monday, after 3:30 PM)

Run the "Scout": Execute the Scout portion of the simulator_htf_scout_sniper.py script.

What it Does: The Scout scans all stocks. It looks for weekly patterns where a breakout on the completed daily chart of Day T has already occurred and all EOD filters were met.

Output: A high-probability "Target List" is generated for the Sniper to monitor on the next trading day, Day T+1.

Pre-Market on Day T+1 (e.g., Tuesday, before 9:15 AM)

Run Imminence Filter (if enabled): This optional filter can be run to refine the Target List.

What it Does: It checks if the daily candle from Day T was an "Inside Day" or "NR7". Only stocks showing this recent volatility contraction are kept on the active Target List for the Sniper.

Intraday during Day T+1 (e.g., Tuesday, 9:15 AM - 3:30 PM)

"Sniper" Monitors for Conviction: The Sniper monitors the 15-minute data for stocks on the active Target List only.

What it Does: It is not waiting for a price breakout (that was confirmed yesterday). It is waiting for a 15-minute candle that shows sufficient conviction (passing Intraday Market Strength and Volume Velocity filters) to confirm the breakout has momentum.

Execute & Manage: If conviction is found, an entry is simulated. Open positions are also managed intraday for exits.