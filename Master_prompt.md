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
The simulator_* scripts are the realistic, actionable versions of the benchmarks. Their sole purpose is to replicate the benchmark logic without lookahead bias, while adding layers of real-world execution logic.

simulator_daily_hybrid.py: This is the primary and most advanced realistic simulator for the daily strategy. Its core bias-free principle is to generate a watchlist at the end of Day T-1 using only data available at that time. On Day T, it then monitors this watchlist for an intraday entry, which must be validated by an advanced conviction and risk engine.

simulator_htf_scout_sniper.py: (No changes needed for this description, it remains accurate.) This uses the "Scout and Sniper" architecture. The Scout runs at the end of Day T to find weekly patterns that have already confirmed a breakout on the completed daily chart of Day T. The Sniper then operates on Day T+1, looking only for intraday conviction signals (not a price breakout).

5.1. The Advanced Conviction & Risk Engine (simulator_daily_hybrid.py)
To improve the quality of entries and model real-world conditions, the hybrid simulator employs a sophisticated conviction and risk engine with the following key components:

Adaptive Slippage Model: Entry prices are adjusted for slippage. The slippage percentage is increased during periods of high market volatility (as measured by the INDIAVIX index).

Dynamic Position Sizing: The size of a new position is calculated based on the available risk capital of the portfolio. It considers the total risk of all currently open positions and will not take a new trade if it would exceed the maximum portfolio risk limit.

Time-Anchored Volume Projection: Instead of a simple volume check, this system projects the end-of-day volume based on the cumulative volume at specific times (e.g., 10:00 AM, 11:30 AM). An entry is only permitted if the volume is "on track" to meet the benchmark's EOD criteria.

Volume Velocity Detection: The volume projection is enhanced with a velocity check. If the script detects a recent surge in 15-minute volume (e.g., using a median-based calculation), it will temporarily relax the projection threshold, allowing it to capture strong momentum breakouts.

VIX-Adaptive Market Strength Filter: The check for broad market strength is made adaptive. During periods of high volatility (high VIX), the threshold for an acceptable market dip is made more lenient, preventing the strategy from becoming overly defensive in choppy markets.

Position Capping: The simulator will not open more than a predefined maximum number of new positions on any given day, preventing over-trading.

Section 6: Strategic Roadmap for Optimization
(This section remains the official path forward for the project.)

Phase 1: Implement Core Risk Architecture (Objective: Control portfolio-level risk)

Phase 2: Sharpen the Alpha & Build Conviction Engine (Objective: Improve the quality of entry signals)

Phase 3: Implement Dynamic Risk (Objective: Link capital allocation to signal quality)

Phase 4: The Final Frontier (Predictive Modeling) (Objective: Bridge the final performance gap)

Section 7: Current Project Status & Immediate Next Steps
(This section has been added to reflect the latest project developments.)

Current Status:
Recent backtesting experiments have yielded a critical insight: attempts to manage risk after a trade has been entered (e.g., with end-of-day reconciliation checks) have not improved performance over the original simulator_daily_hybrid.py. This strongly suggests that the most significant opportunity for improvement lies in enhancing the quality of the initial entry decision, rather than in post-entry management. The original simulator_daily_hybrid.py, with its basic intraday conviction filters, remains the best-performing realistic model to date.

Immediate Next Steps:
Based on these findings, the project will now officially proceed with the "Strategic Roadmap for Optimization." The immediate focus will be on Phase 2: Sharpen the Alpha & Build Conviction Engine. The goal is to make the entry logic of the simulator_daily_hybrid.py more intelligent and robust. The first planned enhancement is the implementation of an Intraday Relative Strength Filter to ensure the strategy is only entering stocks that are demonstrating strength against the market at the moment of breakout.