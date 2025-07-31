Project Code Explained (Version 4.5)
This document provides a detailed functional overview of each key Python script in the Nifty 200 Pullback Strategy project, including its importance, what it accomplishes, and its critical logic as of the latest version.

1. Data Pipeline Scripts
fyers_scrapers (A suite of scripts)
Importance: These are the foundational scripts of the entire project. Without them, no research or backtesting is possible.

Accomplishment: They connect to the data vendor's API to download and save four distinct types of raw historical data: Daily Equity Data, 15-Minute Equity Data, Daily Index Data (NIFTY200, INDIAVIX), and 15-Minute Index Data.

Critical Logic: The core logic involves iterating through a list of symbols, formatting date ranges, and making paginated API calls with robust error handling to manage API rate limits and ensure complete data acquisition over long historical periods.

calculate_indicators_clean.py
Importance: This script is the central data processing engine. It ensures that all backtesting scripts work with a clean, consistent, and pre-calculated dataset, preventing redundant calculations and potential data inconsistencies.

Accomplishment: It reads all the raw daily data for both stocks and indices, resamples it into higher timeframes (weekly, monthly), calculates all necessary technical indicators (EMAs, SMAs, Relative Strength, ATR) for every single timeframe, and saves the enriched dataframes to disk.

Critical Logic: The most critical block is the resample() and agg() logic. It correctly aggregates daily data into weekly candles (using the 'W-FRI' rule to end weeks on Friday) and monthly candles by taking the first open, maximum high, minimum low, last close, and the sum of volume for the period.

2. Benchmark Generators (With Lookahead Bias)
benchmark_generator_daily.py & benchmark_generator_htf.py
Importance: These scripts generate the "Golden Benchmarks" for the daily and HTF strategies, respectively. They represent the maximum theoretical performance of the strategies by intentionally using information that would not be available in real time (i.e., with perfect foresight). They serve as an essential upper bound for validating the performance of the realistic simulators.

Accomplishment: They run the strategies using only End-of-Day (EOD) data. As part of the standardized logging update, they now save their reports to dedicated subdirectories (benchmark_daily/ and benchmark_htf/) for better organization.

Critical Logic (Lookahead Bias): After identifying a setup, these scripts use the completed EOD data from the trigger day to check the Volume and Relative Strength filters. This is a lookahead bias because, in a live intraday scenario, the final EOD volume and closing price are unknown at the moment of entry.

3. Realistic Simulators (Bias-Free)
simulator_daily_hybrid.py
Importance: The advanced, realistic backtesting engine for the Daily Pullback Strategy.

Accomplishment: It simulates the daily strategy's complete lifecycle. It performs a scan after the market closes to find high-quality setups and then monitors those specific setups intraday on the following day for execution, using a sophisticated set of real-time filters. It now saves all its output logs to a dedicated strategy folder (simulator_daily_hybrid/) and implements the full enhanced logging suite.

Critical Logic & Workflow:

EOD Scan (Day T-1): After the market closes, the script scans all stocks for the daily pullback pattern and applies EOD quality filters (Market Regime, Volume, RS) to create a high-probability Watchlist for the next day.

Intraday Monitoring (Day T): On the next trading day, it monitors the 15-minute data stream for only the stocks on the Watchlist.

Advanced Conviction & Risk Engine: Before executing a trade, it validates the breakout against a series of bias-free checks, including Time-Anchored Volume Projection and VIX-Adaptive Market Strength.

htf_simulator_advanced.py (Flagship Weekly Simulator)
Importance: This is the state-of-the-art, bias-free backtester for the HTF (weekly) strategy.

Accomplishment: It uses a "Scout and Sniper" architecture to cleanly separate the process of identifying a weekly setup from the decision to enter a trade. It implements the full suite of advanced conviction filters and saves all its output logs to a dedicated strategy folder (simulator_htf_advanced/).

Critical Logic & Workflow (Scout and Sniper):

Scout Mission (EOD Friday): The Scout runs only once a week, after the market closes on Friday. It scans the weekly data to find valid pullback patterns and generates a "Target List" which is then valid for the entire following week.

Sniper Mission (Intraday, Monday-Friday): The Sniper monitors the 15-minute data stream for only the stocks on the weekly Target List and validates any breakout using the full Advanced Conviction & Risk Engine.

simulator_monthly_advanced.py (New Monthly Simulator)
Importance: This is the new, state-of-the-art simulator for the Monthly Pullback Strategy, adapting the successful Scout/Sniper architecture for a much longer timeframe.

Accomplishment: It simulates the monthly strategy with unique, timeframe-appropriate logic. It implements the full enhanced logging suite, including MAE tracking, and saves all logs to its dedicated folder (simulator_monthly_advanced/).

Critical Logic & Workflow (Monthly Adaptation):

Scout Mission (EOD, Last Trading Day of Month): The Scout runs only once a month. It scans the monthly charts for the pullback pattern and applies monthly-level quality filters (e.g., 10-month EMA, 12-month volume average). It then generates a "Target List" valid for the entire next month.

Sniper Mission (Intraday, Full Month): The Sniper monitors the Target List intraday, but operates within an Adaptive Execution Window. It typically skips the first few volatile days of the month and then actively monitors for a set period (e.g., 15 trading days), which can be extended during high-VIX periods.

Volatility-Adjusted Stop-Loss: A key innovation for this timeframe. The initial stop-loss is not based on a fixed daily lookback. Instead, it is calculated dynamically using the stock's 6-month Average True Range (ATR), providing a volatility-normalized risk buffer appropriate for a long-term trade.

4. Validation & Analysis Scripts
validate_*_subset.py
Importance: These are essential diagnostic tools used to ensure the logical integrity and correctness of the backtesting framework.

Accomplishment: They compare the trade logs from a benchmark run and a simulator run and confirm that the set of trades taken by the simulator is a true and proper subset of the trades taken by its corresponding benchmark.

Critical Logic: The core of the validation uses Python's set.issubset() operation on the unique setup_id generated for each trade in the log files.

mae_analyzer.py (New)
Importance: A powerful, data-driven analysis tool for optimizing stop-loss strategy. It moves beyond simple backtest results to perform a "what-if" simulation on trade outcomes.

Accomplishment: It ingests one or more _trade_details.csv log files and analyzes the Maximum Adverse Excursion (MAE) of each trade. It simulates how the strategy's performance (Win Rate, Profit Factor, P&L) would have changed with a range of different, tighter stop-loss levels.

Critical Logic & Features:

Two Modes of Operation: It can run in a default mode, scanning and analyzing all trade logs in a specified directory, or in a file-specific mode to analyze a single backtest run.

Correlation with Summary: It automatically finds and displays the original performance metrics from the corresponding _summary.txt file, providing crucial context for the analysis.

Holistic Analysis: It provides both a per-file analysis and a combined, holistic analysis of all trades from all supplied files.

Data-Driven Recommendation: Based on the analysis, it generates a plain-English recommendation for an optimal stop-loss range, highlighting the "sweet spot" that best balances cutting losses while preserving the majority of winning trades.