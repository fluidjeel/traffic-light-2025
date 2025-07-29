Public Code repo: https://github.com/fluidjeel/traffic-light-2025

All files are present in the root of the repo. To set deeper context, feel free to ask for any further files if required.

Your final task is to summarize your understanding by directly paraphrasing the rules from the "Definitive Trading Strategy Logic" section and the architectural components described. Do not introduce any external concepts or terminology. Your summary must be a precise, factual reflection of the provided materials, proving you have built a mental model based only on the information given. Once I verify you have the full context, we will work together on the next steps.

this is a private custom strategy refer to github for the documentation and relevant files. Attached are further files for better awareness . same is also found in github

_____Context setting starts____

Master Project Context: Nifty 200 Pullback Strategy (Version 4.0)
Objective: The following document provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. Your goal is to fully ingest this document and the specified files in the correct sequence to build a complete and accurate mental model of the project's architecture, data flow, and the precise logic of its trading strategies. Do not assume any prior knowledge; this document and the files it references are the single source of truth.

Section 1: Context Setting - The Required Reading Sequence
To understand this project, you must study the following files in the precise order listed below. This sequence is designed to build your knowledge from the high-level concept down to the specific implementation details.

project_implementation.md (This Document): Read this file first and in its entirety. It contains the master overview of the project's goals, architecture, and the definitive logic for all trading strategies.

project_code_explained.md: After understanding the master plan, read this file to get a high-level overview of what each specific Python script accomplishes.

project_runbook_operational_guide.md: This document will explain the standard operating procedure for running the entire research pipeline, from data acquisition to analysis.

Core Benchmark & Simulator Scripts: Finally, review the source code of the key scripts.

benchmark_generator_htf.py (The biased, theoretical model)

htf_simulator_advanced.py (The new, state-of-the-art, bias-free simulator)

Section 2: High-Level Project Goal & Philosophy
2.1. Goal
The primary objective of this project is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge. The strategy is applied to the Nifty 200 stock universe.

2.2. Core Philosophy
Our guiding principle is an unwavering commitment to eliminating all forms of bias. All development and testing must rigorously distinguish between "Golden Benchmarks" (biased, theoretical models) and "Realistic Simulators" (bias-free, actionable models). The project prioritizes the identification and mitigation of not only lookahead bias but also systemic flaws like survivorship bias and the impact of transaction costs.

Section 3: System Architecture & Data Pipeline
The project is a modular pipeline designed for rigorous, bias-free research.

Data Acquisition: Raw daily and 15-minute historical data is fetched using a suite of fyers_scrapers.

Data Processing & Resampling: The calculate_indicators_clean.py script acts as the central data processing engine, creating clean, enriched data files for all required timeframes.

Benchmark Generation: The benchmark_generator_* scripts run the strategies with intentional lookahead bias to establish the theoretical "perfect" performance.

Realistic Simulation: The simulator_* scripts, particularly the new htf_simulator_advanced.py, run the strategies in a bias-free manner to replicate real-world trading conditions.

Section 4: The Definitive and Exclusive Trading Strategy Logic
4.1. The HTF Strategy
This strategy identifies a setup on a weekly chart and confirms the entry on the daily chart.

Setup Identification (Weekly):

The most recently completed weekly candle must be a green_candle.

It must be preceded by at least one red_candle on the weekly chart.

It must close above its 30-week EMA.

Daily Breakout Confirmation:

After identifying a valid weekly pattern, it waits for a daily candle's high within the current week to cross above the high of the preceding weekly red candles.

Entry Conditions (on the Day of the Daily Breakout):

The benchmark applies EOD filters (Market Regime, Volume, RS) with lookahead bias. The realistic simulator replaces these with a bias-free intraday conviction engine.

4.2. Universal Trade Management Rules
Initial Stop-Loss: The lowest low of the 5 daily candles preceding the entry day.

Partial Profit Target: A risk/reward target is set. If hit, half the position is sold, and the stop-loss is moved to breakeven.

Trailing Stop-Loss: For the remaining half, the stop is trailed under the low of any subsequent green daily candle.

Section 5: The htf_simulator_advanced.py
This script is the new flagship simulator for the project. It uses a superior "Scout and Sniper" architecture to achieve maximum realism.

Scout (EOD Friday): The Scout runs only at Friday's close to identify valid weekly setups. It pre-calculates the trigger_price and target_daily_volume and adds the stock to a Target List for the entire following week.

Sniper (Intraday Mon-Fri): The Sniper monitors the Target List. When a stock's price crosses its trigger price, it validates the breakout against the Advanced Conviction & Risk Engine. A trade is only executed if all real-time checks pass.

5.1. The Advanced Conviction & Risk Engine
This engine is a suite of bias-free, intraday filters used to validate entry signals.

Time-Anchored Volume Projection: An entry is only permitted if the intraday volume is "on track" to meet the benchmark's criteria and the check occurs after the first time-anchor (10:00 AM).

VIX-Adaptive Market Strength & Slippage: This filter correctly uses the previous day's closing VIX value to avoid lookahead bias when adjusting market strength thresholds and slippage.

Intraday Relative Strength (RS) Filter: This bias-free filter compares the stock's intraday performance against the Nifty 200's intraday performance at the moment of breakout.

Integrated Portfolio Risk Gate: The size of a new position is calculated based on the total available risk capital of the portfolio.

Dynamic Profit Target: The simulator includes a dynamic profit target that sets a more conservative 1:1 RR target in high-VIX markets and a more aggressive 1.5:1 RR target in calm markets.

Section 6: Current Project Status & Immediate Next Steps
Current Status: The project has successfully developed the htf_simulator_advanced.py, a robust, bias-free engine for the HTF strategy. However, initial backtests, while logically sound from a lookahead perspective, have produced unrealistically high returns. Analysis has concluded this is due to two major systemic flaws in the current backtesting environment.

Immediate Next Steps: The project's primary focus has shifted. The goal is no longer to add more entry filters, but to fix the foundational issues that are inflating performance. The next steps are:

Implement a Transaction Cost Model: Integrate a realistic cost model into the simulators to account for brokerage, STT, and other fees.

Mitigate Survivorship Bias: Research and implement a solution for using point-in-time historical index constituents instead of a static nifty200.csv file. This is the highest priority for achieving trustworthy backtest results.

_____Context setting ends
