Master Project Context Prompt: Nifty 200 Pullback Strategy (Version 3.0)
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. Your goal is to fully ingest this document and the specified files in the correct sequence to build a complete and accurate mental model of the project's architecture, data flow, and the precise logic of its trading strategies. Do not assume any prior knowledge; this document and the files it references are the single source of truth.

Section 1: Context Setting - The Required Reading Sequence
To understand this project, you must study the following files in the precise order listed below. This sequence is designed to build your knowledge from the high-level concept down to the specific implementation details.

project_implementation.md (This Document): Read this file first and in its entirety. It contains the master overview of the project's goals, architecture, and the definitive logic for all trading strategies.

project_code_explained.md: After understanding the master plan, read this file to get a high-level overview of what each specific Python script accomplishes.

project_runbook_operational_guide.md: This document will explain the standard operating procedure for running the entire research pipeline, from data acquisition to analysis.

Core Benchmark Scripts: Finally, review the source code of the two "Golden Benchmark" scripts. They are the purest implementation of the trading logic, albeit with lookahead bias.

benchmark_generator_daily.py

benchmark_generator_htf.py

Section 2: High-Level Project Goal & Status
Goal: To develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge on the Nifty 200 stock universe.

Current Status: The project is in an advanced R&D phase. The foundational backtesting framework, consisting of two distinct benchmark generators and two corresponding bias-free simulators, is complete and has been logically validated. The project is now ready to proceed with the strategic roadmap for optimization.

Section 3: System Architecture & Data Pipeline
The project is a modular pipeline designed for rigorous, bias-free research.

Data Acquisition: Raw daily and 15-minute historical data for all stocks and indices is fetched using a suite of fyers_scrapers.

Data Processing & Resampling: The calculate_indicators_clean.py script acts as the central data processing engine. It takes the raw daily data, resamples it into higher timeframes (e.g., 'W-FRI' for weekly), calculates all necessary indicators (EMAs, SMAs, RS, ATR) for all timeframes, and saves these clean, enriched data files.

Benchmark Generation: The benchmark_generator_* scripts run the strategies with intentional lookahead bias to establish the theoretical "perfect" performance.

Realistic Simulation: The simulator_* scripts run the strategies in a bias-free manner, using a combination of EOD and intraday data to replicate real-world trading conditions.

Validation & Analysis: The validate_*_subset.py scripts are used to prove that the simulators are logically aligned with their benchmarks.

Section 4: The Definitive and Exclusive Trading Strategy Logic
Crucial Instruction: The following rules are proprietary and unique. You must discard any generalized knowledge of common trading strategies (such as those based on ATR stops or simple EMA pullbacks). Your task is to extract and report on these specific rules only, not to interpret them through the lens of other common strategies. Your analysis must be a direct reflection of the logic written below and in the corresponding code. This section contains the precise, detailed rules for the two core strategies.

4.1. The Daily Strategy (benchmark_generator_daily.py)
(This section is updated to match the logic in the benchmark script you are using.)

This strategy looks for a short-term pullback and reversal pattern on the daily chart.

Setup Identification (on Day T-1):

Find a Green Candle: The candle for Day T-1 must be a green_candle (close > open).

Apply Quality Filters: This green candle must meet a strict quality criterion:

It must close above its 30-day EMA.
(Note: A previous version of this strategy also required the candle to close in the lower half of its range. This filter has been removed to make the setup criteria less strict.)

Confirm a Preceding Pullback: The script looks back from Day T-2 to confirm at least one preceding red_candle.

Entry Conditions (on Day T):

Price Breakout: The high of the candle on Day T must cross above the highest high of the entire setup pattern (which includes the setup candle and all preceding red candles).

Market Regime Filter: The Nifty 200 Index must be trading above its 50-day EMA.

Volume Filter: The volume on Day T must be at least 1.3x its 20-day average volume.

Relative Strength (RS) Filter: The stock's 30-day price return must be greater than the Nifty 200's 30-day price return.

4.2. The HTF Strategy (benchmark_generator_htf.py)
This strategy identifies a setup on a higher timeframe (e.g., weekly) and confirms the entry on the daily chart.

Setup Identification (Multi-Timeframe):

Find a Weekly Pattern: It looks at the most recently completed weekly candle. It checks for the following:

The weekly candle must be a green_candle.

It must be preceded by at least one red_candle on the weekly chart.

It must close above its 30-week EMA.

It must close in the lower half of its weekly range (close < (high + low) / 2).

Confirm a Daily Breakout: After identifying a valid weekly pattern, it waits for a daily candle within the current week to meet these conditions:

The daily candle's high must cross above the high of the preceding weekly red candles.

It must be the first time this breakout has happened within the current weekly period.

Entry Conditions (on the Day of the Daily Breakout):

On the same day the daily breakout is confirmed, it applies the exact same three End-of-Day filters as the daily strategy (Market Regime, Volume, RS).

4.3. Universal Trade Management Rules
Once a position is opened in either strategy, it is managed by the following rules:

Initial Stop-Loss: The lowest low of the 5 daily candles preceding the entry.

Partial Profit Target: A 1:1 risk/reward target is set. If hit, half the position is sold, and the stop-loss is moved to breakeven.

Trailing Stop-Loss: For the remaining half, the stop is trailed under the low of any subsequent green daily candle.

Section 5: The Bias-Free Simulators
(This section is updated to be more precise about the simulator's logic.)

The simulator_* scripts are the realistic, actionable versions of the benchmarks. Their sole purpose is to replicate the benchmark logic without lookahead bias.

simulator_daily_hybrid.py: This is the primary realistic simulator for the daily strategy. At the end of Day T-1, it identifies the complete setup pattern. It then immediately applies the benchmark's End-of-Day filters (Market Regime, Volume, and RS) using only the data available at the close of Day T-1. If a stock passes all checks, it is added to a watchlist for Day T, where it is monitored for an intraday price breakout confirmed by real-time conviction filters.

simulator_htf_scout_sniper.py: (No changes needed for this description, it remains accurate.) This uses the "Scout and Sniper" architecture. The Scout runs at the end of Day T to find weekly patterns that have already confirmed a breakout on the completed daily chart of Day T. The Sniper then operates on Day T+1, looking only for intraday conviction signals (not a price breakout).