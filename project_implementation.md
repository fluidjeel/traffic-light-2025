Master Project Context Prompt: Nifty 200 Pullback Strategy (Updated)
Objective: The following prompt provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. The goal is for you, the LLM, to fully understand the project's current architecture, its development history, the core research challenge it faces, and the logic of its key backtesting models. Do not assume any prior knowledge; this document is the single source of truth.

1. High-Level Project Goal & Status
1.1. Goal:
The project's objective is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge. The target is to achieve a high CAGR while maintaining acceptable drawdown levels.

1.2. Current Status:
The project is in an advanced research and development phase. The backtesting framework is now composed of two distinct, bias-free hybrid models: one for daily setups (final_backtester_v8_hybrid_optimized.py) and one for higher-timeframe setups (final_backtester_htf_hybrid.py). Both have been enhanced with a suite of toggleable, experimental features. The current focus is on systematically testing these features to find the optimal configuration and then proceeding with the strategic roadmap for further enhancements.

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

5. Future Vision: The Refined Strategic Roadmap
With the backtesting framework now robust and bias-free for all intended timeframes, the project will now proceed with the formal strategic roadmap. The focus is on implementing the features from Phase 1 and 2 of the roadmap to systematically improve CAGR and reduce drawdown.

Phase 1: Implement Core Risk Architecture & Diagnostics
Objective: Immediately control portfolio-level risk and establish the tools needed for data-driven optimization.

Implement Portfolio-Level Circuit Breakers:

Max Open Positions: Code a hard limit on concurrent open positions (e.g., 3-5).

Daily Drawdown Limit: Implement a portfolio-wide "kill switch" that halts new entries for the day if the daily P&L drops below a set threshold (e.g., -2.5% of equity).

Implement Systematic Performance Diagnostics: Build the diagnostic logger to compare Golden Benchmark trades to the Hybrid model's actions, identifying key sources of performance leakage.

Phase 2: Sharpen the Alpha & Build Conviction Engine
Objective: Improve the quality of entry signals and create the logic engine for dynamic risk.

Implement Adaptive Volume Velocity Filter: Integrate the filter that compares current 15-min volume to the 70th percentile for that specific time slot, with a multiplier that adapts to the VIX.

Implement Intraday Relative Strength Compass: Integrate the filter that calculates the real-time performance spread against the Nifty 200 from the pre-market high.

Develop the Composite Signal Conviction Score: Create a function that generates a single numerical score representing the quality of a setup based on all active filters.

Phase 3: Implement Dynamic Risk & Advanced Filters
Objective: Link capital allocation directly to signal quality and add further layers of confirmation.

Implement Dynamic Position Sizing: Replace the fixed-risk model. Use the Composite Signal Conviction Score from Phase 2 to dynamically adjust the percentage of capital risked on each trade.

Add Advanced Confirmation Filters:

Multi-Timeframe (MTF) Confirmation: Add the Daily -> Hourly -> 15-Min alignment check.

Market Breadth Filter: Add the real-time Advance/Decline ratio check.

Phase 4: The Final Frontier (Predictive Modeling)
Objective: Bridge the final performance gap to the benchmark using advanced, data-driven techniques.

Develop the Predictive ML Filter: Train a GradientBoostingClassifier to predict the probability of a setup passing the benchmark's EOD filters.

Develop "Second Chance" Protocol: Code and test the protocol to intelligently re-enter valid setups that were missed on the initial breakout.

6. Operational Timeline: A Day-by-Day Breakdown
This section details the precise sequence of events for both bias-free backtesting models to ensure a clear understanding of the operational flow and the prevention of lookahead bias.

Daily Hybrid Strategy (final_backtester_v8_hybrid_optimized.py)
T-1, Post-Market (e.g., Monday Evening):

Action: The script performs a fast scan on the completed daily charts from Monday (T-1).

Logic: It identifies all stocks that formed a valid "red candles + green candle" pattern.

Output: A Watchlist is generated for Tuesday.

Day T, Pre-Market (e.g., Tuesday Morning, before 9:15 AM):

Action: The script checks the opening tick/pre-market data for stocks on the Watchlist.

Logic: If the cancel_on_gap_up feature is enabled, any stock set to open above its trigger price is removed from the Watchlist.

Day T, During Trading Hours (e.g., Tuesday, 9:15 AM - 3:30 PM):

Action: The script monitors the 15-minute chart for all stocks remaining on the Watchlist.

Logic: It waits for a 15-minute candle's high to cross the trigger price. Once triggered, it then waits for a 15-minute candle to close that satisfies all real-time conviction filters (Volume Velocity, Market Strength, etc.).

Output: Trades are executed. Open positions are managed (intraday stop-loss checks, partial profit exits).

Day T, Post-Market (e.g., Tuesday Evening):

Action: The script performs end-of-day management for all open positions.

Logic: It checks the completed daily candle for Tuesday to apply trailing stop logic (Aggressive Breakeven, trailing on green candle low).

Output: Stop-loss levels for Wednesday are updated. The cycle then repeats for the next day.

HTF "Scout and Sniper" Hybrid Strategy (final_backtester_htf_hybrid.py)
T-1, Post-Market (e.g., Monday Evening):

Action: The "Scout" runs its daily EOD scan.

Logic:

It checks the completed weekly chart to find stocks in a valid pullback pattern (one or more red candles).

For those stocks, it checks the completed daily chart from Monday (T-1) to see if a breakout above the weekly trigger price occurred.

Output: If both conditions are met, the stock is added to a Target List for Tuesday.

Day T, Pre-Market (e.g., Tuesday Morning, before 9:15 AM):

Action: The script performs a check on the Target List.

Logic: (This model does not use the cancel_on_gap_up feature, as the breakout has already been confirmed. The entry is based on follow-through conviction).

Day T, During Trading Hours (e.g., Tuesday, 9:15 AM - 3:30 PM):

Action: The "Sniper" monitors the 15-minute chart for stocks on the Target List only.

Logic: It waits for a 15-minute candle to close that satisfies all real-time conviction filters (Volume Velocity, Market Strength, etc.). The price breakout has already been confirmed by the Scout.

Output: Trades are executed. Open positions are managed.

Day T, Post-Market (e.g., Tuesday Evening):

Action: The script performs EOD management for open positions and runs the "Scout" again.

Logic:

Applies trailing stop logic to all open positions based on Tuesday's completed daily candle.

The "Scout" runs its scan on Tuesday's EOD data to generate a new Target List for Wednesday.

Output: Stop-loss levels are updated, and a new Target List is created. The cycle repeats.