Master Project Context Prompt: Nifty 200 Pullback Strategy (Updated)
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. The goal is for you, the LLM, to fully understand the project's current architecture, its development history, the core research challenge it faces, and the logic of its key backtesting models. Do not assume any prior knowledge; this document is the single source of truth.

1. High-Level Project Goal & Status
1.1. Goal:
The project's objective is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge. The target is to achieve a high CAGR while maintaining acceptable drawdown levels.

1.2. Current Status:
The project is in a highly advanced research and development phase. The initial development led to the discovery of a critical lookahead bias in the original backtesting model, which produced unrealistically high returns.

The project has since pivoted to a more rigorous, professional approach. The current focus is on building and validating a realistic Hybrid Intraday Model (final_backtester_v8_hybrid_optimized.py) that is free of lookahead bias, and analyzing its performance against a "perfect" (but flawed) benchmark to quantify the true cost of real-world execution.

2. The Core Research Challenge: Lookahead Bias vs. Reality
The central theme of this project's development is the journey from a flawed backtest to a realistic one.

The Flawed Model ("Golden Benchmark"): The original backtester made entry decisions intraday (Day T) but used filter data (total daily volume, daily closing prices) that was only available at the end of Day T. This "crystal ball" gave it an impossible advantage, resulting in an exceptionally high CAGR (~50-60%). We now use a dedicated script (final_backtester_benchmark_logger.py) to run this flawed logic intentionally, creating a "perfect" trade log that serves as our Golden Benchmark.

The Realistic Model (Hybrid Intraday): The current state-of-the-art model (final_backtester_v8_hybrid_optimized.py) was built to eliminate this bias. It attempts to capture the same setups identified by the benchmark, but by using a realistic, hybrid daily/intraday approach that relies only on information available at the moment of a trade decision.

The primary research task is to analyze the performance gap between the Golden Benchmark and the Realistic Model and to enhance the Realistic Model to bridge this gap through intelligent, data-driven improvements.

3. System Architecture & Script Workflow
The project is a modular data pipeline. The current key scripts are:

3.1. Data Pipeline:

Data Scrapers: fyers_equity_scraper.py (for daily data), fyers_equity_scraper_15min.py (for intraday equity data), and fyers_nifty200_index_scraper_15min.py (for intraday index data).

Indicator Calculator (calculate_indicators_clean.py): The definitive data processing engine. It takes raw daily data, aggregates it into all required timeframes (2-day, weekly, monthly), correctly calculates all necessary indicators (EMAs, RS, etc.), and saves the clean, processed files.

3.2. Backtesting Engines:

Benchmark Logger (final_backtester_benchmark_logger.py): Runs the original, flawed strategy with lookahead bias. Its sole purpose is to generate the "Golden Benchmark" log file (_all_setups_log.csv), which contains every "perfect" trade the system could find with future knowledge. It also tracks the hypothetical PnL of every setup.

Hybrid Backtester (final_backtester_v8_hybrid_optimized.py): The current, most advanced, and realistic backtesting engine. It is the primary tool for current research.

4. Core Trading Logic: A Tale of Two Models
4.1. Universal Trade Management (Applies to Both Models):
The exit logic is consistent across all models to ensure a fair comparison.

Initial Stop-Loss: The lowest low of the 5 daily candles preceding the entry day.

Two-Leg Exit Strategy:

Leg 1 (Partial Profit): 50% of the position is sold at a 1:1 risk/reward target.

Move to Breakeven: The stop-loss for the remaining shares is moved to the entry price.

Leg 2 (Trailing Stop): The stop is trailed daily to the higher of the breakeven price or the low of the most recent green daily candle.

4.2. "Golden Benchmark" Entry Logic (Lookahead Bias):

Timeframe: Daily.

Setup: On Day T, it identifies a price action pattern (red candles + green candle) that completed on the previous day (T-1).

Execution: It checks if the price on Day T broke out above the trigger price.

The Flaw: It then confirms this breakout by checking filters (Market Regime, Volume, RS) using the final, end-of-day data from Day T.

4.3. "Hybrid Intraday" Entry Logic (Realistic):
This model uses a hybrid daily/intraday approach to avoid lookahead bias.

Strategic Setup (Daily): At the start of each day, it scans the completed daily chart from the previous day (T-1) to find the price action setup. This generates a "watchlist" for the day.

Tactical Execution (15-Minute Chart): It then loops through the 15-minute candles of the current day for stocks on the watchlist.

Confirmation & Entry: A trade is entered at the close of the first 15-minute candle where all the filter conditions (Volume Velocity, Intraday Market Strength, etc.) have been met in real-time, provided the price has not run away beyond a predefined slippage limit.

5. Future Vision: Drawdown Control & Agentic AI
The current research focus is on improving the performance of the realistic Hybrid Model. The next planned steps are:

Drawdown Control: Implementing risk management layers (e.g., limiting the number of open positions, dynamic risk allocation) to control the strategy's drawdowns.

Agentic AI: Exploring the use of LLMs and agentic frameworks to enhance the strategy with more dynamic, intelligent decision-making (e.g., catalyst scanning, adaptive trade management).
