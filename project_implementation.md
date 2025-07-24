Master Project Context Prompt: Nifty 200 Pullback Strategy (Updated)
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. The goal is for you, the LLM, to fully understand the project's current architecture, its development history, the core research challenge it faces, and the logic of its key backtesting models. Do not assume any prior knowledge; this document is the single source of truth.

1. High-Level Project Goal & Status
1.1. Goal:
The project's objective is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge. The target is to achieve a high CAGR while maintaining acceptable drawdown levels.

1.2. Current Status:
The project is in a highly advanced research and development phase. The initial development led to the discovery of a critical lookahead bias in the original backtesting model.

The project has since pivoted to a more rigorous, professional approach. The current focus is on enhancing and validating the realistic Hybrid Intraday Model (final_backtester_v8_hybrid_optimized.py), which is now equipped with a suite of experimental, toggleable filters and trade management rules. The primary research task is to analyze the impact of these new features and systematically improve the strategy's performance against a truthful, risk-adjusted baseline.

2. The Core Research Challenge: Lookahead Bias vs. Reality
The central theme of this project's development is the journey from a flawed backtest to a realistic one.

The Flawed Model ("Golden Benchmark"): The original backtester made entry decisions intraday (Day T) but used filter data (total daily volume, daily closing prices) that was only available at the end of Day T. This "crystal ball" gave it an impossible advantage. We now use a dedicated script (final_backtester_benchmark_logger.py) to run this flawed logic intentionally, creating a "perfect" trade log that serves as our Golden Benchmark.

The Realistic Model (Hybrid Intraday): The current state-of-the-art model (final_backtester_v8_hybrid_optimized.py) was built to eliminate this bias. It attempts to capture the same setups identified by the benchmark, but by using a realistic, hybrid daily/intraday approach that relies only on information available at the moment of a trade decision.

3. System Architecture & Script Workflow
The project is a modular data pipeline. The current key scripts are:

3.1. Data Pipeline:
Data Scrapers: fyers_equity_scraper.py (for daily equity data), fyers_index_scraper.py (for daily index data like VIX), and fyers_equity_scraper_15min.py (for intraday equity data).

Indicator Calculator (calculate_indicators_clean.py): The definitive data processing engine. It takes raw daily data, calculates all necessary indicators, and saves the clean, processed files.

3.2. Backtesting Engines:
Benchmark Logger (final_backtester_benchmark_logger.py): Runs the original, flawed strategy with lookahead bias to generate the "Golden Benchmark" log file.

Hybrid Backtester (final_backtester_v8_hybrid_optimized.py): The current, most advanced, and realistic backtesting engine, now featuring multiple experimental toggles. It is the primary tool for current research.

4. Core Trading Logic: A Tale of Two Models
4.1. Universal Trade Management (Applies to Both Models):
The exit logic is consistent in principle but has been enhanced in the realistic model with experimental rules.

Initial Stop-Loss: The lowest low of the 5 daily candles preceding the entry day.

Two-Leg Exit Strategy (Toggleable):

Leg 1 (Partial Profit): 50% of the position is sold at a 1:1 risk/reward target. This feature can now be disabled to let the entire position run.

Move to Breakeven: The stop-loss for the remaining shares is moved to the entry price.

Trailing Stop Logic:

Standard Trail: The stop is trailed daily to the higher of the breakeven price or the low of the most recent green daily candle.

Aggressive Breakeven (Toggleable): An experimental rule that moves the stop to a few ticks above the entry price as soon as a trade is profitable at the end of the day, aiming to prevent a winner from turning into a loser.

4.2. "Golden Benchmark" Entry Logic (Lookahead Bias):
Setup: On Day T, it identifies a price action pattern that completed on the previous day (T-1).

Execution: It checks if the price on Day T broke out above the trigger price.

The Flaw: It then confirms this breakout by checking filters using the final, end-of-day data from Day T.

4.3. "Hybrid Intraday" Entry Logic (Realistic):
This model uses a hybrid daily/intraday approach to avoid lookahead bias, now enhanced with several new filters.

Strategic Setup (Daily): At the start of each day, it scans the completed daily chart from the previous day (T-1) to find the price action setup. This generates a "watchlist" for the day.

Pre-Market Check (Toggleable): Before the market opens, it checks if any watchlist stocks are set to open above the trigger price (cancel_on_gap_up) and cancels them.

Tactical Execution (15-Minute Chart): It then loops through the 15-minute candles of the current day for stocks on the watchlist.

Confirmation & Entry: A trade is entered at the close of the first 15-minute candle where all filter conditions are met in real-time. This now includes:

A check to ensure the price has not reversed below the trigger price after confirmation (prevent_entry_below_trigger).

A realistic fill price simulation using a Dynamic Slippage Model based on liquidity and VIX.

5. Future Vision: Data-Driven Optimization
The project is now in a phase of rigorous, data-driven experimentation. The focus is on analyzing the impact of the newly implemented features to determine the optimal combination of rules for maximizing CAGR and minimizing drawdown. The next planned steps involve systematically testing each toggleable feature and using the generated logs to inform the "Refined Strategic Roadmap."