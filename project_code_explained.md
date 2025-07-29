Project Code Explained (Version 4.0)
This document provides a detailed functional overview of each key Python script in the Nifty 200 Pullback Strategy project, including its importance, what it accomplishes, and its critical logic as of the latest version.

1. Data Pipeline Scripts
fyers_scrapers (A suite of scripts)
Importance: These are the foundational scripts of the entire project. Without them, no research or backtesting is possible.

Accomplishment: They connect to the data vendor's API to download and save four distinct types of raw historical data: Daily Equity Data, 15-Minute Equity Data, Daily Index Data (NIFTY200, INDIAVIX), and 15-Minute Index Data.

Critical Logic: The core logic involves iterating through a list of symbols, formatting date ranges, and making paginated API calls with robust error handling.

calculate_indicators_clean.py
Importance: This script is the central data processing engine. It ensures that all backtesting scripts work with a clean, consistent, and pre-calculated dataset.

Accomplishment: It reads all the raw daily data for both stocks and indices, resamples it into higher timeframes (weekly, monthly), calculates all necessary technical indicators (EMAs, SMAs, RS, ATR) for every timeframe, and saves the enriched dataframes.

Critical Logic: The most critical block is the resample() and agg() logic. It correctly aggregates daily data into weekly candles (e.g., using 'W-FRI') by taking the first open, max high, min low, last close, and sum of volume.

2. Benchmark Generators (With Lookahead Bias)
benchmark_generator_daily.py
Importance: Generates the "Golden Benchmark" for the daily strategy, representing the maximum theoretical performance with perfect foresight.

Accomplishment: Runs the daily strategy with intentional lookahead bias.

Critical Logic: After identifying a setup on Day T-1, it uses the completed EOD data from Day T to check the Volume and RS filters, which is not possible in a real-world intraday context.

benchmark_generator_htf.py
Importance: Generates the "Golden Benchmark" for the HTF strategy.

Accomplishment: Runs the HTF strategy with lookahead bias.

Critical Logic: On the day of a daily breakout, it uses the completed daily candle's data to apply the Volume and RS filters, which is a form of lookahead bias.

3. Realistic Simulators (Bias-Free)
simulator_daily_hybrid-v2.py
Importance: The advanced, realistic backtesting engine for the daily strategy.

Accomplishment: Identifies a setup on Day T-1 and attempts to execute it on Day T using 15-minute intraday data and a sophisticated set of real-time, bias-free filters.

Critical Logic: Its bias-free design is centered on its Advanced Conviction & Risk Engine, which validates entries using only data available at the moment of the breakout (e.g., volume projection, intraday RS, previous day's VIX).

htf_simulator_advanced.py (Flagship HTF Simulator)
Importance: This is the new state-of-the-art, bias-free backtester for the HTF strategy, replacing all previous versions.

Accomplishment: It uses a superior "Scout and Sniper" architecture to separate the discovery of a weekly setup from the decision to enter a trade, ensuring maximum realism.

Critical Logic:

Scout (EOD Friday): The Scout runs only at Friday's close to identify valid weekly setups. It pre-calculates the trigger price and target volume for the entire next week.

Sniper (Intraday Mon-Fri): The Sniper monitors the target list. When a breakout occurs, it validates the entry using the full Advanced Conviction & Risk Engine. This includes critical bias-free checks, such as using the previous day's VIX close for volatility adjustments and ensuring volume projection checks only pass after the first time-based threshold (10:00 AM).

4. Validation & Analysis Scripts
validate_*_subset.py
Importance: Essential diagnostic tools to ensure the logical integrity of the backtesting framework.

Accomplishment: Mathematically prove that a simulator's trade universe is a true subset of its benchmark's universe.

Critical Logic: Uses Python's set.issubset() operation on the setup_id from the log files to perform the validation.