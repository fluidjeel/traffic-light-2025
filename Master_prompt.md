Master Project Context: Nifty 200 Pullback Strategy (Version 4.4)
To the Large Language Model:

This is a private, custom algorithmic trading strategy. The information contained in the attached documents and code files is the single source of truth. Do not reference any external code, public GitHub repositories, or general trading knowledge. Your sole objective is to build a complete and accurate mental model of this project based only on the provided materials.

Public GitHub Repo for context: https://github.com/fluidjeel/traffic-light-2025. All files are available at the root of the repo.

Your tasks are as follows:

Ingest Context: Fully ingest the context provided below and in the specified file sequence.

Prove Foundational Understanding: Your first response must be a direct paraphrase of the rules from the "Definitive Trading Strategy Logic" section and a summary of the key architectural components. Do not introduce any external concepts.

Prove Deeper Understanding: After your initial summary, you must answer the following questions to prove you have understood the relationship between the system's components:

Question 1: In htf_simulator_advanced.py (the weekly simulator), if a researcher disables the use_vix_adaptive_filters by setting it to False, which specific component's behavior changes: the Scout's, the Sniper's, or both? Explain your reasoning by referencing the logic described in the documentation.

Question 2: In simulator_monthly_advanced.py, explain the precise mathematical relationship between the atr_multiplier_stop and the risk_per_trade_percent when calculating position size. If a researcher makes the stop-loss wider by increasing atr_multiplier_stop from 1.2 to 1.8, what is the direct, calculated impact on the number of shares purchased for a given trade, assuming all other factors remain constant?

Context Setting Starts
Objective:
The following document provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. Your goal is to fully ingest this document and the specified files in the correct sequence to build a complete and accurate mental model of the project's architecture, data flow, and the precise logic of its trading strategies.

Section 1: Context Setting - The Required Reading Sequence
To understand this project, you must study the following files in the precise order listed below. This sequence is designed to build your knowledge from the high-level concept down to the specific implementation details.

project_implementation.md: Read this file first and in its entirety. It contains the master overview of the project's goals, architecture, and the definitive logic for all trading strategies.

project_code_explained.md: After understanding the master plan, read this file to get a high-level overview of what each specific Python script accomplishes.

project_runbook_operational_guide.md: This document will explain the standard operating procedure for running the entire research pipeline, from data acquisition to analysis.

Core Simulator & Analysis Scripts: Finally, review the source code of the key scripts to understand the implementation details.

simulator_daily_hybrid.py (The daily timeframe simulator)

htf_simulator_advanced.py (The weekly timeframe simulator)

simulator_monthly_advanced.py (The new monthly timeframe simulator)

mae_analyzer.py (The new stop-loss analysis tool)

Section 2: High-Level Project Goal & Philosophy
2.1. Goal
The primary objective is to develop, backtest, and automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge, applied to the Nifty 200 stock universe across multiple timeframes.

2.2. Core Philosophy
Our guiding principle is an unwavering commitment to eliminating all forms of bias. All development and testing must rigorously distinguish between "Golden Benchmarks" (biased, theoretical models) and "Realistic Simulators" (bias-free, actionable models).

Section 3: System Architecture & Data Pipeline
The project is a modular pipeline designed for rigorous, bias-free research.

Data Acquisition: Raw historical data is fetched using a suite of fyers_scrapers.

Data Processing: The calculate_indicators_clean.py script creates enriched data files for all required timeframes (daily, weekly, monthly).

Realistic Simulation: The simulator_* scripts run the strategies in a bias-free manner to replicate real-world trading.

Standardized Logging: All simulators save their output logs to dedicated subdirectories within backtest_logs, controlled by the strategy_name key in their configuration. The logs include detailed trade data, including Maximum Adverse Excursion (MAE) for advanced analysis.

Post-Hoc Analysis: The mae_analyzer.py script ingests trade logs to perform "what-if" simulations for stop-loss optimization.

Section 4: The Definitive and Exclusive Trading Strategy Logic
4.1. The Monthly Strategy
This strategy identifies a setup on a monthly chart and confirms the entry on a daily/intraday chart.

Setup Identification (Monthly):

The most recently completed monthly candle must be a green_candle.

It must be preceded by at least one red_candle on the monthly chart.

(Optional Filter) It should close above its 10-month EMA.

Daily Breakout Confirmation:

After identifying a valid monthly pattern, it waits for a daily candle's high within the subsequent month to cross above the high of the monthly pattern.

Trade Management:

Initial Stop-Loss: A volatility-adjusted stop is used, calculated as entry_price - (6_month_ATR * atr_multiplier).

Profit Target: A dynamic R:R target is used (e.g., 2.0R in calm markets, 1.5R in volatile markets).

Trailing Stop: The stop is trailed under the low of subsequent green daily candles.

Section 5: The simulator_monthly_advanced.py
This script is the new flagship simulator for the monthly strategy. It uses the "Scout and Sniper" architecture.

Scout (EOD, Last Day of Month): The Scout runs only at month-end to identify valid monthly setups. It pre-calculates the trigger_price and the monthly_atr and adds the stock to a Target List for the entire next month.

Sniper (Intraday, Full Month): The Sniper monitors the Target List. It operates within an Adaptive Execution Window, skipping the first few days of the month and then monitoring for a set period. When a stock's price crosses its trigger, it executes the trade.

Section 6: Current Project Status & Immediate Next Steps
Current Status: The project has successfully developed a suite of robust, bias-free simulators for daily, weekly, and monthly timeframes, complete with a standardized and detailed logging system capable of advanced MAE analysis. The immediate focus remains on addressing foundational backtesting flaws.

Immediate Next Steps:

Implement a Transaction Cost Model: Integrate a realistic cost model into all simulators.

Mitigate Survivorship Bias: Implement a solution for using point-in-time historical index constituents.

Context Setting Ends