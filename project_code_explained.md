Project Code Explained (Version 5.0)
This document provides a detailed functional overview of each key Python script in the modern Traffic Light 2025 project, reflecting the new universal data pipeline and the final, corrected simulators.

1. Universal Data Pipeline Scripts
universal_fyers_scraper.py

Importance: This is the single, foundational data acquisition script for the entire project.

Accomplishment: It connects to the Fyers API to download and save all required raw historical data (Daily and 15-Minute for both equities and indices). It features intelligent, incremental updates to only fetch new data, a --force flag for full re-downloads, and robust error handling. It saves all data to a unified data/universal_historical_data/ directory.

Critical Logic: The core logic involves iterating through a symbol list, making paginated API calls, and checking for existing data to determine the correct date range for fetching updates.

universal_calculate_indicators.py

Importance: This is the central data processing engine for the new pipeline.

Accomplishment: It reads the raw daily data from the universal scraper's output folder, resamples it into all higher timeframes (weekly, monthly), calculates all necessary technical indicators (EMAs, SMAs, ATRs, etc.), and saves the enriched dataframes to a unified data/universal_processed/ directory.

Critical Logic: The most critical block is the resample() and agg() logic that correctly aggregates daily data into higher timeframe candles before applying the indicator calculations.

2. Realistic & Flexible Simulators (Bias-Free)
The project now uses a suite of three flexible, fully-corrected simulators. Each is capable of running on either the new universal data pipeline or the legacy data folders by changing a single toggle in its configuration.

daily_tfl_simulator.py

Importance: The bias-free backtesting engine for the Daily Pullback Strategy.

Accomplishment: It simulates the daily strategy's complete lifecycle. It performs an EOD scan to find high-quality setups and then monitors those setups intraday on the following day for execution, using a sophisticated set of real-time conviction filters.

Critical Logic: The script is built on a two-day cycle: EOD scan on Day T-1 to generate a watchlist, followed by intraday monitoring and execution on Day T. All calculations (position sizing, VIX timing, etc.) have been corrected to be free of lookahead bias and logical flaws.

weekly_tfl_simulator.py

Importance: The state-of-the-art, bias-free backtester for the HTF (weekly) strategy.

Accomplishment: It uses a "Scout and Sniper" architecture to cleanly separate the process of identifying a weekly setup from the decision to enter a trade. It implements the full suite of advanced conviction filters.

Critical Logic (Scout and Sniper): The "Scout" runs EOD on Friday to generate a "Target List" for the entire following week. The "Sniper" then monitors this list intraday from Monday to Friday, validating any breakouts with the Conviction Engine. All calculations have been corrected for integrity.

monthly_tfl_simulator.py

Importance: The state-of-the-art simulator for the Monthly Pullback Strategy.

Accomplishment: It simulates the monthly strategy by adapting the Scout/Sniper model for a longer timeframe. It includes unique, timeframe-appropriate logic like an adaptive execution window and VIX-scaled profit targets.

Critical Logic (Monthly Adaptation): The "Scout" runs EOD on the last trading day of the month to generate a Target List valid for the entire next month. The "Sniper" monitors this list intraday during the following month. All calculations have been corrected for integrity.

3. Analysis Scripts
mae_analyzer_percent.py

Importance: A powerful, data-driven analysis tool for optimizing the stop-loss strategy of the percentage-based simulators.

Accomplishment: It ingests one or more _trade_details.csv log files and analyzes the Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE) of each trade. It runs "what-if" simulations to show how performance would change with different fixed percentage stop-losses.

Critical Logic: The core logic involves reading trade logs, calculating MAE/MFE percentages, and then iterating through a range of hypothetical stop-loss percentages to re-calculate performance metrics like Win Rate and Profit Factor for the trades that would have survived.