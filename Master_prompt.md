Master Project Context Prompt: Nifty 200 Pullback Strategy
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. The goal is for you, the LLM, to fully understand the project's architecture, data flow, and the precise logic of the two distinct trading strategies to assist with future modifications, bug fixes, or the addition of new features. Do not assume any prior knowledge; this document is the single source of truth.

1. High-Level Project Goal & Status
1.1. Goal:
The project's objective is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge.

1.2. Current Status:
The project is in an advanced research and development phase. The backtesting framework is now composed of two distinct, bias-free hybrid models: one for daily setups (final_backtester_v8_hybrid_optimized.py) and one for higher-timeframe setups (final_backtester_htf_hybrid.py). Both have been enhanced with a suite of toggleable, experimental features. The current focus is on systematically testing these features to find the optimal configuration and then proceeding with the strategic roadmap for further enhancements.

2. System Architecture & Script Workflow
The project is a modular data pipeline. Each script performs a distinct, specialized task.

2.1. Script Pipeline:
Data Acquisition:

fyers_equity_scraper.py: Downloads daily equity data.

fyers_index_scraper.py: Downloads daily index data (Nifty 200, India VIX).

fyers_equity_scraper_15min.py: Downloads 15-minute equity data.

Indicator Calculation (calculate_indicators_clean.py): Takes raw daily data, calculates all necessary indicators for all timeframes (daily, weekly, monthly), and saves the processed files.

Strategy Simulation (Two Primary Engines):

final_backtester_v8_hybrid_optimized.py: The bias-free backtester for the daily timeframe strategy.

final_backtester_htf_hybrid.py: The bias-free backtester for the higher timeframe (weekly-immediate, monthly-immediate) strategies, built on the "Scout and Sniper" logic.

Benchmark Generation (Two Engines):

final_backtester_benchmark_logger.py: Generates the "Golden Benchmark" for the daily strategy.

final_backtester_immediate_benchmark_htf.py: Generates the "Golden Benchmark" for the HTF-immediate strategy.

Post-Hoc Analysis (analyze_breakeven_impact.py): A dedicated script to analyze the net P&L impact of the "Aggressive Breakeven" feature.

3. Core Trading Strategies: Detailed Logic
3.1. Daily Hybrid Strategy (final_backtester_v8_hybrid_optimized.py)
Setup (T-1, EOD): Identifies a pattern of red candles followed by a green candle on the completed daily chart.

Execution (Day T, Intraday): Monitors the 15-minute chart. Enters on the first candle that crosses the trigger price and meets all real-time conviction filters (Volume Velocity, etc.).

3.2. HTF "Scout and Sniper" Hybrid Strategy (final_backtester_htf_hybrid.py)
Scout (Day T, EOD):

Identifies a valid pullback pattern (one or more red candles) on the completed weekly chart (T-1).

Checks the just-completed daily candle (Day T) to see if it confirmed a breakout above the weekly pattern's high.

If both are true, it adds the stock to a target list for tomorrow (Day T+1).

Sniper (Day T+1, Intraday):

Monitors the 15-minute chart for stocks on the target list only.

Enters on the first 15-minute candle that meets all real-time conviction filters.

3.3. Universal Trade Management & Experimental Features
Both bias-free backtesters share the same suite of toggleable features for execution and trade management:

Dynamic Slippage Model: Simulates realistic transaction costs based on liquidity (volume) and volatility (VIX).

Partial Profit Target: A toggleable 1:1 risk/reward partial profit exit.

Aggressive Breakeven: A toggleable rule to move the stop to just above breakeven on profitable EOD positions.

HTF Trailing Stop: A toggleable rule (for the HTF backtester) to switch to a weekly-low-based trailing stop after a trade is profitable.