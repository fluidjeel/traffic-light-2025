Master Project Context Prompt: Nifty 200 Pullback Strategy (Updated)
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. The goal is for you, the LLM, to fully understand the project's current architecture, its development history, the core research challenge it faces, and the logic of its key backtesting models. Do not assume any prior knowledge; this document is the single source of truth.

1. High-Level Project Goal & Status
1.1. Goal:
The project's objective is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge. The target is to achieve a high CAGR while maintaining acceptable drawdown levels.

1.2. Current Status:
The project is in a highly advanced research and development phase. The backtesting framework is now composed of two distinct, bias-free hybrid models: one for daily setups (final_backtester_v8_hybrid_optimized.py) and one for higher-timeframe setups (final_backtester_htf_hybrid.py). Both have been enhanced with a suite of toggleable, experimental features. The current focus is on systematically testing these features to find the optimal configuration and then proceeding with the strategic roadmap for further enhancements.

2. The Core Research Challenge: Lookahead Bias vs. Reality
The central theme of this project's development is the journey from a flawed backtest to a realistic one, which has now been achieved for both daily and higher-timeframe strategies.

The Flawed Models ("Golden Benchmarks"): These are legacy backtesters that use end-of-day data to make intraday decisions, giving them an impossible "crystal ball" advantage. We use final_backtester_benchmark_logger.py for the daily strategy and final_backtester_immediate_benchmark_htf.py for the HTF strategy to generate these "perfect" trade logs.

The Realistic Models (Bias-Free):

Daily Hybrid (final_backtester_v8_hybrid_optimized.py): Identifies a setup on Day T-1 and seeks to execute it on Day T using only real-time intraday data.

HTF "Scout and Sniper" Hybrid (final_backtester_htf_hybrid.py): A more advanced, from-scratch model that cleanly separates discovery from execution. It uses completed EOD data on Day T to find a breakout of a weekly pattern (the "Scout") and then seeks to execute on Day T+1 using real-time intraday data (the "Sniper"). This architecture guarantees the elimination of lookahead bias.

3. System Architecture & Script Workflow
The project is a modular data pipeline. The current key scripts are:

3.1. Data Pipeline:
Data Scrapers: fyers_equity_scraper.py, fyers_index_scraper.py (for VIX), and fyers_equity_scraper_15min.py.

Indicator Calculator (calculate_indicators_clean.py): The definitive data processing engine that calculates all indicators for all timeframes.

3.2. Backtesting Engines:
Benchmark Loggers: final_backtester_benchmark_logger.py (Daily) and final_backtester_immediate_benchmark_htf.py (HTF).

Hybrid Backtesters: final_backtester_v8_hybrid_optimized.py (Daily) and final_backtester_htf_hybrid.py (HTF).

3.3. Analysis Scripts:
analyze_breakeven_impact.py: A dedicated script to quantify the net P&L impact of the "Aggressive Breakeven" feature.

4. Core Trading Logic & Experimental Features
Both bias-free backtesters share a common suite of advanced, toggleable features for execution and trade management:

Dynamic Slippage Model: Simulates realistic transaction costs based on liquidity (volume) and volatility (VIX).

Partial Profit Target: A toggleable 1:1 risk/reward partial profit exit.

Aggressive Breakeven: A toggleable rule to move the stop to just above breakeven on profitable EOD positions.

HTF Trailing Stop: A toggleable rule (for the HTF backtester) to switch to a weekly-low-based trailing stop after a trade is profitable.

5. Future Vision: The Strategic Roadmap
With the backtesting framework now robust and bias-free for all intended timeframes, the project will now proceed with the formal "Refined Strategic Roadmap." The focus is on implementing the features from Phase 1 and 2 of the roadmap, such as portfolio-level risk controls (e.g., max open positions) and more advanced alpha filters (e.g., adaptive volume).