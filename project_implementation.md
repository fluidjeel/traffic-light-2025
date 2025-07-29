Master Project Context: Nifty 200 Pullback Strategy (Version 4.0)
Objective: The following document provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. Your goal is to fully ingest this document and the specified files in the correct sequence to build a complete and accurate mental model of the project's architecture, data flow, and the precise logic of its trading strategies. Do not assume any prior knowledge; this document and the files it references are the single source of truth.

Section 1: High-Level Project Goal & Philosophy
1.1. Goal
The primary objective of this project is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge. The strategy is applied to the Nifty 200 stock universe.

1.2. Core Philosophy
Our guiding principle is an unwavering commitment to eliminating all forms of bias. All development and testing must rigorously distinguish between "Golden Benchmarks" (biased, theoretical models that represent the strategy's maximum potential) and "Realistic Simulators" (bias-free, actionable models that replicate real-world trading conditions). The project prioritizes the identification and mitigation of not only lookahead bias but also systemic flaws like survivorship bias and the impact of transaction costs.

Section 2: System Architecture & Data Pipeline
The project is a modular pipeline designed for rigorous, bias-free research.

Data Acquisition: Raw daily and 15-minute historical data for all stocks and indices is fetched using a suite of fyers_scrapers.

Data Processing & Resampling: The calculate_indicators_clean.py script acts as the central data processing engine. It takes the raw daily data, resamples it into higher timeframes (e.g., 'W-FRI' for weekly), calculates all necessary indicators (EMAs, SMAs, RS, ATR) for all timeframes, and saves these clean, enriched data files.

Benchmark Generation: The benchmark_generator_* scripts run the strategies with intentional lookahead bias to establish the theoretical "perfect" performance.

Realistic Simulation: The simulator_* scripts run the strategies in a bias-free manner, using a combination of EOD and intraday data to replicate real-world trading conditions.

Validation & Analysis: The validate_*_subset.py scripts are used to prove that the simulators are logically aligned with their benchmarks.

Section 3: The Definitive and Exclusive Trading Strategy Logic
3.1. The Daily Strategy (benchmark_generator_daily.py)
This strategy looks for a short-term pullback and reversal pattern on the daily chart.

Setup Identification (on Day T-1):

The candle for Day T-1 must be a green_candle (close > open).

It must close above its 30-day EMA.

The script looks back from Day T-2 to confirm at least one preceding red_candle.

Entry Conditions (on Day T):

The high of the candle on Day T must cross above the highest high of the entire setup pattern.

Market Regime Filter: The Nifty 200 Index must be trading above its 50-day EMA.

Volume Filter: The volume on Day T must be at least 1.3x its 20-day average volume.

Relative Strength (RS) Filter: The stock's 30-day price return must be greater than the Nifty 200's 30-day price return.

3.2. The HTF Strategy (benchmark_generator_htf.py)
This strategy identifies a setup on a weekly chart and confirms the entry on the daily chart.

Setup Identification (Weekly):

The most recently completed weekly candle must be a green_candle.

It must be preceded by at least one red_candle on the weekly chart.

It must close above its 30-week EMA.

Daily Breakout Confirmation:

After identifying a valid weekly pattern, it waits for a daily candle's high within the current week to cross above the high of the preceding weekly red candles.

It must be the first time this breakout has happened within the current weekly period.

Entry Conditions (on the Day of the Daily Breakout):

On the same day the daily breakout is confirmed, it applies the exact same three End-of-Day filters as the daily strategy (Market Regime, Volume, RS).

3.3. Universal Trade Management Rules
Initial Stop-Loss: The lowest low of the 5 daily candles preceding the entry day.

Partial Profit Target: A risk/reward target is set. If hit, half the position is sold, and the stop-loss is moved to breakeven.

Trailing Stop-Loss: For the remaining half, the stop is trailed under the low of any subsequent green daily candle.

Section 4: The Bias-Free Simulators
4.1. simulator_daily_hybrid-v2.py (Advanced Daily Simulator)
This is the state-of-the-art simulator for the daily strategy. It identifies a complete setup pattern at the close of Day T-1 and on Day T, it monitors for an intraday price breakout confirmed by a sophisticated, bias-free conviction engine.

4.2. htf_simulator_advanced.py (Flagship HTF Simulator)
This script is a complete rewrite and the new standard for HTF backtesting, replacing the older simulator_htf_scout_sniper.py. It uses a superior "Scout and Sniper" architecture to achieve maximum realism.

Scout (EOD Friday): The Scout runs only at the close of the market on Friday. It scans all tradable stocks to find valid weekly setups according to the definitive strategy rules. For each valid setup, it pre-calculates the trigger_price and target_daily_volume and adds the stock to a Target List for the entire following week.

Sniper (Intraday Mon-Fri): The Sniper monitors the Target List intraday. When a stock's price crosses its trigger price, it does not enter immediately. Instead, it validates the breakout against the full Advanced Conviction & Risk Engine. A trade is only executed if all real-time checks pass.

4.3. The Advanced Conviction & Risk Engine
This engine is a core component of both advanced simulators (daily and htf). It is a suite of bias-free, intraday filters used to validate entry signals.

Time-Anchored Volume Projection: Instead of using future EOD volume, this system projects the final volume based on cumulative volume at specific times (e.g., 10:00 AM, 11:30 AM). An entry is only permitted if the volume is "on track" to meet the benchmark's criteria and the check occurs after the first time-anchor (10:00 AM).

VIX-Adaptive Market Strength & Slippage: The check for broad market strength (Nifty 200's intraday performance) is made adaptive. During periods of high volatility, the threshold for an acceptable market dip is made more lenient. This filter correctly uses the previous day's closing VIX value to avoid lookahead bias. Slippage is also increased in high-VIX environments.

Intraday Relative Strength (RS) Filter: This bias-free filter compares the stock's intraday performance (since the open) against the Nifty 200's intraday performance at the moment of breakout.

Integrated Portfolio Risk Gate: The size of a new position is calculated based on the available risk capital of the portfolio. It considers the total risk of all currently open positions and will not take a new trade if it would exceed the maximum portfolio risk limit (e.g., 6% of equity).

Dynamic Profit Target (HTF Only): The HTF simulator includes a dynamic profit target that sets a more conservative 1:1 RR target in high-VIX markets and a more aggressive 1.5:1 RR target in calm markets.

Section 5: Strategic Roadmap & Current Status
5.1. Strategic Roadmap
Phase 0: Foundational Validation & Refinement (Complete)

Phase 1: Implement Core Risk Architecture (Complete)

Phase 2: Sharpen the Alpha & Build Conviction Engine (Complete)

Phase 3: Address Systemic Backtesting Flaws (In Progress)

Objective: Eliminate systemic biases to produce truly realistic performance metrics.

Phase 4: The Final Frontier (Predictive Modeling)

5.2. Current Project Status & Immediate Next Steps
Current Status: The project has successfully developed the htf_simulator_advanced.py, a robust, bias-free engine for the HTF strategy. However, initial backtests, while logically sound from a lookahead perspective, have produced unrealistically high returns. Analysis has concluded this is due to two major systemic flaws in the current backtesting environment.

Immediate Next Steps: The project's primary focus has shifted to Phase 3. The goal is no longer to add more entry filters, but to fix the foundational issues that are inflating performance. The next steps are:

Implement a Transaction Cost Model: Integrate a realistic cost model into the simulators to account for brokerage, STT, and other fees.

Mitigate Survivorship Bias: Research and implement a solution for using point-in-time historical index constituents instead of a static nifty200.csv file. This is the highest priority for achieving trustworthy backtest results.