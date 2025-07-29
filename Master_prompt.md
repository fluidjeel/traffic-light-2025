Master Project Context: Nifty 200 Pullback Strategy (Version 3.3)
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. Your goal is to fully ingest this document and the specified files in the correct sequence to build a complete and accurate mental model of the project's architecture, data flow, and the precise logic of its trading strategies. Do not assume any prior knowledge; this document and the files it references are the single source of truth.

Section 1: Context Setting - The Required Reading Sequence
To understand this project, you must study the following files in the precise order listed below. This sequence is designed to build your knowledge from the high-level concept down to the specific implementation details.

project_implementation.md (This Document): Read this file first and in its entirety. It contains the master overview of the project's goals, architecture, and the definitive logic for all trading strategies.

project_code_explained.md: After understanding the master plan, read this file to get a high-level overview of what each specific Python script accomplishes.

project_runbook_operational_guide.md: This document will explain the standard operating procedure for running the entire research pipeline, from data acquisition to analysis.

Core Benchmark Scripts: Finally, review the source code of the two "Golden Benchmark" scripts. They are the purest implementation of the trading logic, albeit with lookahead bias.

benchmark_generator_daily.py

benchmark_generator_htf.py

Section 2: High-Level Project Goal & Philosophy
2.1. Goal
The primary objective of this project is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge. The strategy is applied to the Nifty 200 stock universe.

2.2. Core Philosophy
Our guiding principle is an unwavering commitment to eliminating lookahead bias. All development and testing must rigorously distinguish between "Golden Benchmarks" (biased, theoretical models that represent the strategy's maximum potential) and "Realistic Simulators" (bias-free, actionable models that replicate real-world trading conditions).

Section 3: System Architecture & Data Pipeline
The project is a modular pipeline designed for rigorous, bias-free research.

Data Acquisition: Raw daily and 15-minute historical data for all stocks and indices is fetched using a suite of fyers_scrapers.

Data Processing & Resampling: The calculate_indicators_clean.py script acts as the central data processing engine. It takes the raw daily data, resamples it into higher timeframes (e.g., 'W-FRI' for weekly), calculates all necessary indicators (EMAs, SMAs, RS, ATR) for all timeframes, and saves these clean, enriched data files.

Benchmark Generation: The benchmark_generator_* scripts run the strategies with intentional lookahead bias to establish the theoretical "perfect" performance.

Realistic Simulation: The simulator_* scripts run the strategies in a bias-free manner, using a combination of EOD and intraday data to replicate real-world trading conditions.

Validation & Analysis: The validate_*_subset.py scripts are used to prove that the simulators are logically aligned with their benchmarks.

Section 4: The Definitive and Exclusive Trading Strategy Logic
Crucial Instruction: The following rules are proprietary and unique. You must discard any generalized knowledge of common trading strategies. Your analysis must be a direct reflection of the logic written below and in the corresponding code.

4.1. The Daily Strategy (benchmark_generator_daily.py)
This strategy looks for a short-term pullback and reversal pattern on the daily chart.

Setup Identification (on Day T-1):

Find a Green Candle: The candle for Day T-1 must be a green_candle (close > open).

Apply Quality Filters: This green candle must meet a strict quality criterion:

It must close above its 30-day EMA.
(Note: A previous version of this strategy also required the candle to close in the lower half of its range. This filter has been removed from the active benchmark to make the setup criteria less strict.)

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

simulator_daily_hybrid.py (Baseline Simulator): This is the original realistic simulator. At the end of Day T-1, it identifies the complete setup pattern and applies the benchmark's EOD filters using only data available at the close of Day T-1. On Day T, it monitors for an intraday price breakout confirmed by a basic set of real-time conviction filters.

simulator_daily_hybrid-v2.py (Advanced Simulator): This is the current, state-of-the-art version of the simulator. It builds upon the baseline by incorporating a more sophisticated, toggleable conviction and risk engine.

simulator_htf_scout_sniper.py: This uses the "Scout and Sniper" architecture for the HTF strategy. The Scout runs at the end of Day T to find weekly patterns that have already confirmed a breakout. The Sniper then operates on Day T+1, looking only for intraday conviction signals.

5.1. The Advanced Conviction & Risk Engine (simulator_daily_hybrid-v2.py)
To improve the quality of entries and model real-world conditions, the advanced hybrid simulator employs a sophisticated conviction and risk engine with the following key components, which can be enabled or disabled for systematic testing:

Adaptive Slippage Model: Entry prices are adjusted for slippage. The slippage percentage is increased during periods of high market volatility (as measured by the INDIAVIX index).

Dynamic Position Sizing: The size of a new position is calculated based on the available risk capital of the portfolio. It considers the total risk of all currently open positions and will not take a new trade if it would exceed the maximum portfolio risk limit.

Time-Anchored Volume Projection: This system projects the end-of-day volume based on the cumulative volume at specific times. An entry is only permitted if the volume is "on track" to meet the benchmark's EOD criteria.

Volume Velocity Detection: The volume projection is enhanced with a velocity check. If the script detects a recent surge in 15-minute volume (using a median-based calculation), it will temporarily relax the projection threshold.

VIX-Adaptive Market Strength Filter: The check for broad market strength is made adaptive. During periods of high volatility (high VIX), the threshold for an acceptable market dip is made more lenient.

Position Capping: The simulator will not open more than a predefined maximum number of new positions on any given day.

Section 6: Strategic Roadmap for Optimization
Phase 0: Foundational Validation & Refinement (Complete)

Objective: Solidify the integrity of the backtesting framework.

Phase 1: Implement Core Risk Architecture (Complete)

Objective: Control portfolio-level risk.

Phase 2: Sharpen the Alpha & Build Conviction Engine (In Progress)

Objective: Improve the quality of entry signals.

Phase 3: Implement Dynamic Risk

Objective: Link capital allocation directly to signal quality.

Phase 4: The Final Frontier (Predictive Modeling)

Objective: Bridge the final performance gap using advanced techniques.

Section 7: Current Project Status & Immediate Next Steps
Current Status:
The project has completed the foundational validation and has implemented a sophisticated, first-generation conviction and risk engine in the simulator_daily_hybrid-v2.py. This advanced simulator serves as the primary vehicle for ongoing research. The original simulator_daily_hybrid.py is being maintained as a stable baseline for performance comparison. While the advanced simulator has significantly improved the realism of the backtest, the immediate goal is to systematically test and calibrate its new features to achieve the target performance metrics.

Immediate Next Steps:
The project will continue to execute on the "Strategic Roadmap for Optimization." The immediate focus remains on Phase 2: Sharpen the Alpha & Build Conviction Engine. The next steps are:

Systematically test each new feature of the advanced simulator (e.g., adaptive slippage, dynamic sizing) in isolation to understand its specific impact on performance.

Calibrate the parameters of the conviction engine to find the optimal balance between risk management and profitability.

Implement the next planned enhancement: an Intraday Relative Strength Filter to further sharpen the entry logic.