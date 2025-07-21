Master Project Context Prompt: Nifty 200 Pullback Strategy
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. The goal is for you, the LLM, to fully understand the project's architecture, data flow, and the precise logic of the two distinct trading strategies to assist with future modifications, bug fixes, or the addition of new features. Do not assume any prior knowledge; this document is the single source of truth.

1. High-Level Project Goal & Status
1.1. Goal:
The project's objective is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy.

1.2. Current Status:
The project is in an advanced stage. The local backtesting framework is complete, stable, and has been thoroughly debugged. It consists of a full pipeline of Python scripts for data management and strategy simulation. The focus has now shifted from local development to planning the migration to a fully automated, cloud-based execution system on AWS.

2. System Architecture & Script Workflow
The project is a modular data pipeline. Each script performs a distinct, specialized task and prepares data for the next script in the chain. The workflow is sequential and must be followed in the specified order.

2.1. Script Pipeline:

Data Acquisition (fyers_..._scraper.py scripts): These scripts are the gateway to the market. Their sole responsibility is to connect to the Fyers API and ensure the raw daily OHLCV data in the historical_data/ folder is complete and up-to-date. They perform efficient incremental updates.

Indicator Calculation (calculate_indicators.py): This script is the data factory. It takes the raw daily data as input, transforms it by creating different timeframes (2-day, weekly, etc.), enriches it by calculating all the technical indicators required by the strategy, and saves the processed files.

Strategy Simulation (Two Engines):

final_backtester.py: The primary engine for simulating the strategy with standard, end-of-period logic.

final_backtester_immediate.py: A specialized engine for simulating an accelerated, "fast entry" strategy on higher timeframes.

Post-Hoc Analysis (analyze_missed_trades.py): An optional final step to analyze the performance of trades that were identified but not taken due to capital constraints.

2.2. Directory & File Structure:

/ (Root Project Directory)

historical_data/: Stores the raw, unprocessed daily OHLCV data.

data/processed/: Parent directory for processed data, organized by timeframe.

daily/, 2day/, weekly/, monthly/

backtest_logs/: All output files from the backtesters are saved here.

config.py: Stores private Fyers API credentials.

fyers_access_token.txt: A temporary session file for the active authentication token.

nifty200.csv: The master list of stock symbols defining the trading universe.

Python Scripts (*.py): The executable core of the project.

3. Core Trading Strategy: Detailed Routines
This section defines the precise, unambiguous logic of the trading system.

3.1. Universal Rules & Filters (Applied by Both Backtesters)
A potential trade setup must pass all of the following filters in sequence to be considered for entry.

Market Regime Filter:

Question: Is the overall market in an uptrend?

Logic: Checks if the daily Nifty 200 index close is above its 50-day EMA. If not, no new long trades are considered for that day.

Relative Strength Filter:

Question: Is this stock stronger than the overall market?

Logic: Checks if the stock's 30-day percentage return is greater than the Nifty 200's 30-day return.

Volume Filter:

Question: Is there conviction behind this breakout?

Logic: Checks if the volume on the entry day is at least 1.3 times its 20-day average.

ATR Filter (Disabled by default):

Question: Is the stock's volatility stable?

Logic: Checks if the short-term volatility (14-day ATR) is not excessively higher than its long-term average (30-day MA of ATR).

3.2. Standard Strategy (final_backtester.py)
Timeframes: daily, 2day, weekly, monthly.

Entry Logic (End-of-Period):

A trade is only considered on the closing day of a period (e.g., on Friday for a weekly timeframe).

The entry decision is based on a two-period pattern of completed candles for that timeframe.

Setup (on Period T-1):

The candle must be green (close > open).

The close must be in the upper 50% of the candle's range.

The close must be above the 30-period EMA.

It must be preceded by one or more consecutive red candles.

Trigger (on Period T):

Trigger Price: The highest high of the entire setup pattern (red candles + green candle).

Execution: A trade is entered if the high of Period T is >= the Trigger Price AND the open of Period T is < the Trigger Price.

3.3. Immediate Entry Strategy (final_backtester_immediate.py)
Timeframes: weekly-immediate, monthly-immediate.

Core Principle: To enter a trade as early as possible within a forming weekly or monthly candle, rather than waiting for it to close.

Entry Logic (Hybrid Timeframe):

HTF Setup: The script identifies a valid setup pattern on the completed higher-timeframe (HTF) charts (e.g., last week's candles), using the same rules as the standard strategy.

Fast Entry Trigger Price: The trigger price is calculated as the highest high of the completed red HTF candles only.

Daily Scan: The script's main loop runs daily. On each day of the current, forming HTF candle, it checks the daily price action.

Execution: A trade is entered on the first day that the daily high crosses above the "Fast Entry Trigger Price," provided it has not already been breached on a previous day within the current HTF period.

3.4. Universal Position & Trade Management (Apply to Both Backtesters)
Position Sizing: Risks a fixed percentage of the current total portfolio equity on each trade. Equity is recalculated after all exits for the day but before new entries are considered.

Initial Stop-Loss: The lowest low of the 5 periods (days, weeks, etc.) immediately preceding the entry period. For the "immediate" strategy, this is based on the daily chart.

Trade Management (Two-Leg Exit Strategy):

Leg 1 (Partial Profit): 50% of the position is sold when the price hits a 1:1 risk/reward target. The stop-loss for the remaining shares is immediately moved to the entry price (breakeven).

Leg 2 (Trailing Stop): For any profitable trade, the stop-loss is trailed daily. The new stop is set to the higher of either the breakeven price or the low of the most recent green candle.

Final Exit: The entire remaining position is closed if the price hits the current stop-loss level. Exits are always checked on a daily basis, even for higher timeframe strategies, to ensure realism.

4. Future Vision: AWS Automation Architecture
The project is planned to be migrated to a fully automated, serverless architecture on AWS.

4.1. Guiding Principles:

Serverless: Use AWS Lambda to avoid managing servers.

Cost-Effective: Pay only for compute time used.

Reliable & Low-Maintenance: Use managed AWS services.

Secure: Use AWS Secrets Manager for all credentials.

4.2. Lambda Function Breakdown:

Authentication_Lambda: A scheduled function to handle the automated browser login to Fyers to generate and securely store the daily access token.

Data_Pipeline_Lambda: A scheduled function to run the scraper and indicator calculation scripts, keeping all data in S3 up-to-date.

Signal_Generation_Lambda: Chained to the data pipeline, this function runs the backtester logic to generate a trade_plan.json file, which it saves to S3.

Order_Execution_Lambda: Triggered by the creation of the trade_plan.json file, this function reads the plan and places the actual orders via the Fyers API. It includes a "PAPER" trading mode for forward testing.

Paper_Trade_Journaling_Lambda: A scheduled function that runs daily to manage and journal the results of the paper trading mode, providing a fully automated forward testing loop.

4.3. Known Limitations to be Addressed:

Manual Authentication: The current local system requires manual intervention for API authentication.

No Corporate Action Handling: The system is not yet designed to handle stock splits or special dividends, which would corrupt historical price data if not adjusted for.

Execution Model: The backtester assumes perfect fills and does not yet model for slippage or price gaps.

5. Supporting Project Documentation
In addition to this master context prompt, the project is supported by a suite of detailed documents available in the project's GitHub repository (https://github.com/fluidjeel/traffic-light-2025). These documents provide deeper dives into specific aspects of the project and should be considered part of the overall context.

project_implementation.md: This file contains the master context prompt itself. It is the primary document for setting the overall project context for an LLM.

project_runbook_operational_guide.md: This is the Standard Operating Procedure (SOP) for the project. It provides a verbose, step-by-step guide on how to run the entire backtesting pipeline from data acquisition to final analysis. It also includes a detailed checklist for manually auditing trades to verify the backtester's integrity. This document is intended for any human operator of the system.

project_code_explained.md: This document serves as a code commentary and guide. It breaks down each key Python script, explaining the functional purpose and programming best practices behind the main functions and logical blocks. It is designed for developers or users who want to understand how the code works at a deeper level.

planned_aws_techstack.md: This document outlines the architectural blueprint for cloud deployment. It details the recommended AWS services (Lambda, S3, EventBridge, etc.), the proposed workflow for a fully automated system, and the rationale behind the chosen serverless architecture. It serves as the plan for migrating the project from a local research environment to a live, automated trading system.