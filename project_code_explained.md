Project Code Explained (Version 3.3)
This document provides a detailed functional overview of each key Python script in the Nifty 200 Pullback Strategy project, including its importance, what it accomplishes, and its critical logic.

1. Data Pipeline Scripts
fyers_scrapers (A suite of scripts)

Importance: These are the foundational scripts of the entire project. Without them, no research or backtesting is possible.

Accomplishment: They connect to the data vendor's API to download and save four distinct types of raw historical data:

Daily Equity Data (fyers_equity_scraper.py)

15-Minute Equity Data (fyers_equity_scraper_15min.py)

Daily Index Data (fyers_index_scraper.py for NIFTY200, INDIAVIX, etc.)

15-Minute Index Data (fyers_nifty200_index_scraper_15min.py)

Critical Logic: The core logic involves iterating through the nifty200.csv list or a predefined list of indices, formatting date ranges, and making paginated API calls with robust error handling.

calculate_indicators_clean.py

Importance: This script is the central data processing engine. It ensures that all backtesting scripts work with a clean, consistent, and pre-calculated dataset.

Accomplishment: It reads all the raw daily data for both stocks and indices, resamples it into higher timeframes (weekly, monthly), calculates all necessary technical indicators (EMAs, SMAs, RS, ATR) for every timeframe, and saves the enriched dataframes.

Critical Logic: The most critical block is the resample() and agg() logic. It correctly aggregates daily data into weekly candles (e.g., using 'W-FRI') by taking the first open, max high, min low, last close, and sum of volume.

2. Benchmark Generators (Lookahead Bias)
benchmark_generator_daily.py

Importance: Generates the "Golden Benchmark" for the daily strategy, representing the maximum theoretical performance.

Accomplishment: Runs the daily strategy with intentional lookahead bias.

Critical Logic: The core of its logic involves two key aspects:

Lookahead Bias: After identifying a setup on Day T-1, it uses the completed data from Day T to check the Volume and RS filters.

On-the-Fly Indicators: It loads raw data, truncates it to the backtest date range, and then calculates all indicators. This "post-truncation" method is a critical detail that must be replicated for validation.

benchmark_generator_htf.py

Importance: Generates the "Golden Benchmark" for the HTF strategy.

Accomplishment: Runs the HTF strategy with lookahead bias.

Critical Logic: On the day of a daily breakout, it uses the completed daily candle's data to apply the Volume and RS filters, which is not possible in real-time.

3. Realistic Simulators (Bias-Free)
simulator_daily_hybrid.py

Importance: The state-of-the-art, realistic backtesting engine for the daily strategy, featuring an advanced conviction and risk engine.

Accomplishment: Identifies a setup on Day T-1 and attempts to execute it on Day T using 15-minute intraday data and a sophisticated set of real-time filters.

Critical Logic: Its bias-free design is multi-faceted:

Watchlist Generation: At the end of Day T-1, it identifies a setup pattern and applies all the benchmark's EOD filters (Volume, RS, Market Regime) using only the data available up to the close of Day T-1.

Advanced Conviction Engine: On Day T, it does not enter on a simple price cross. Instead, it validates the entry with a sophisticated engine that includes:

Adaptive Slippage and Position Sizing: Models realistic execution costs and manages portfolio-level risk.

Volume Projection & Velocity: Uses time-anchored checks and median-based surge detection to ensure strong volume momentum.

VIX-Adaptive Market Strength: Adjusts its market filter based on volatility to avoid being too timid in choppy markets.

simulator_htf_scout_sniper.py

Importance: Eliminates a subtle but powerful form of lookahead bias present in HTF strategies.

Accomplishment: Uses the "Scout and Sniper" architecture to separate the discovery of a confirmed breakout from the decision to enter a trade.

Critical Logic: The Scout (EOD) scans for weekly patterns where a breakout has already occurred on the completed daily chart of Day T. The Sniper (Intraday) then operates on Day T+1, waiting only for intraday conviction signals (e.g., volume velocity) to confirm momentum.

4. Validation & Analysis Scripts
validate_*_subset.py

Importance: Essential diagnostic tools to ensure the logical integrity of the backtesting framework.

Accomplishment: Mathematically prove that a simulator's trade universe is a true subset of its benchmark's universe.

Critical Logic: Uses Python's set.issubset() operation on the setup_id from the log files to perform the validation.

analyze_hybrid_rogue_trades.py

Importance: A specialized analysis tool to understand the performance differences between the benchmark and the hybrid simulator.

Accomplishment: It isolates the trades that are unique to the simulator_daily_hybrid.py (i.e., "rogue trades") and calculates their specific performance contribution.

Critical Logic: It consumes the log files from both the benchmark and the simulator, uses set operations to find the unique setup_ids, and then calculates performance metrics on only those trades.