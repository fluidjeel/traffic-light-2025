Master Project Context: Nifty 200 Pullback Strategy (Version 4.5)
Objective: The following document provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. Your goal is to fully ingest this document and the specified files in the correct sequence to build a complete and accurate mental model of the project's architecture, data flow, and the precise logic of its trading strategies. Do not assume any prior knowledge; this document and the files it references are the single source of truth.

Section 1: High-Level Project Goal & Philosophy
1.1. Goal
The primary objective of this project is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge. The strategy is applied to the Nifty 200 stock universe across multiple timeframes.

1.2. Core Philosophy
Our guiding principle is an unwavering commitment to eliminating all forms of bias. All development and testing must rigorously distinguish between "Golden Benchmarks" (biased, theoretical models) and "Realistic Simulators" (bias-free, actionable models). The project prioritizes the identification and mitigation of not only lookahead bias but also systemic flaws like survivorship bias and the impact of transaction costs.

Section 2: System Architecture & Data Pipeline
The project is a modular pipeline designed for rigorous, bias-free research.

Data Acquisition: Raw daily and 15-minute historical data for all stocks and indices is fetched using a suite of fyers_scrapers.

Data Processing & Resampling: The calculate_indicators_clean.py script acts as the central data processing engine. It takes the raw daily data, resamples it into higher timeframes (weekly, monthly), calculates all necessary indicators (EMAs, SMAs, RS, ATR) for all timeframes, and saves these clean, enriched data files.

Benchmark Generation: The benchmark_generator_* scripts run the strategies with intentional lookahead bias to establish the theoretical "perfect" performance.

Realistic Simulation: The simulator_* scripts run the strategies in a bias-free manner, using a combination of EOD and intraday data to replicate real-world trading conditions.

Validation & Analysis: The validate_*_subset.py scripts are used to prove that the simulators are logically aligned with their benchmarks. A new mae_analyzer.py script has been developed for advanced, data-driven stop-loss optimization.

2.1. Standardized Logging Architecture
A key enhancement has been implemented across all backtesting scripts. Each script now automatically creates a dedicated subdirectory within the backtest_logs folder (e.g., backtest_logs/simulator_monthly_advanced/). This is controlled by the strategy_name key in each script's config dictionary. The realistic simulators have been further enhanced with a toggleable, granular logging system that can produce detailed reports on filled trades, missed trades, and setups rejected by intraday filters. This includes logging for Maximum Adverse Excursion (MAE) and saving the entire configuration dictionary with each summary report to ensure full reproducibility.

Section 3: The Definitive and Exclusive Trading Strategy Logic
3.1. The Daily Strategy (as defined in benchmark_generator_daily.py)
This strategy looks for a short-term pullback and reversal pattern on the daily chart.

Setup Identification (on Day T-1):

The candle for Day T-1 must be a green_candle (close > open).

It must close above its 30-day EMA.

The script looks back from Day T-2 to confirm at least one preceding red_candle.

Entry Conditions (on Day T):

The high of the candle on Day T must cross above the highest high of the entire setup pattern (the green candle and its preceding red candles).

Market Regime Filter: The Nifty 200 Index must be trading above its 50-day EMA.

Volume Filter: The volume on Day T must be at least 1.3x its 20-day average volume.

Relative Strength (RS) Filter: The stock's 30-day price return must be greater than the Nifty 200's 30-day price return.

3.2. The HTF (Weekly) Strategy (as defined in benchmark_generator_htf.py)
This strategy identifies a setup on a weekly chart and confirms the entry on the daily chart.

Setup Identification (Weekly):

The most recently completed weekly candle must be a green_candle.

It must be preceded by at least one red_candle on the weekly chart.

It must close above its 30-week EMA.

Daily Breakout Confirmation:

After identifying a valid weekly pattern, the system waits for a daily candle's high within the current week to cross above the high of the preceding weekly red candles.

It must be the first time this breakout has happened within the current weekly period.

Entry Conditions (on the Day of the Daily Breakout):

On the same day the daily breakout is confirmed, it applies the exact same three End-of-Day filters as the daily strategy (Market Regime, Volume, RS).

3.3. The Monthly Strategy (as defined in simulator_monthly_advanced.py)
This strategy adapts the core pullback concept to a much longer timeframe.

Setup Identification (Monthly):

The most recently completed monthly candle must be a green_candle.

It must be preceded by at least one red_candle on the monthly chart.

(Optional Filter) The monthly candle's close should be above its 10-month EMA.

Daily Breakout Confirmation:

After identifying a valid monthly pattern, the system waits for a daily candle's high within the subsequent month to cross above the high of the monthly pattern.

3.4. Universal Trade Management Rules (Enhanced & Configurable)
The simulators now feature a highly flexible, toggle-driven system for managing trades.

Initial Stop-Loss: The initial risk is defined by the stop_loss_mode config parameter:

LOOKBACK mode: Sets the stop at the lowest low of a specified number of preceding daily candles (e.g., 5 days for the weekly strategy).

PERCENT mode: Sets the stop at a fixed percentage below the entry price (e.g., 9%).

ATR mode (Monthly only): Sets the stop at a multiple of the 6-month Average True Range below the entry price.

Profit Target & Exit Strategy: The overall exit logic is controlled by the exit_strategy_mode config parameter:

TRAILING mode (Default for HTF): This is a complex, multi-stage exit strategy.

An initial profit target is set based on a Risk:Reward multiple of the initial stop-loss.

If use_partial_profit_leg is True, half the position is sold when this target is hit.

The stop-loss for the remaining position is then moved to breakeven and subsequently trailed under the low of any new green daily candle.

ATR_TARGET mode (New option for HTF): This is a simpler, binary exit strategy.

A single profit target is calculated based on a multiple of the weekly ATR (e.g., entry_price + (3 * weekly_atr)).

The trade is exited in full if either the initial stop-loss or this ATR-based profit target is hit. All trailing logic is disabled in this mode.

Aggressive Breakeven: An optional rule that moves the stop-loss to just above the entry price as soon as a trade is profitable.

Section 4: Strategy Timeline and Execution Flow
This section describes the practical, time-based sequence of events for each strategy as implemented in the bias-free simulators.

4.1. The Daily Strategy (simulator_daily_hybrid.py)
This strategy operates on a two-day cycle: setup identification on Day T-1 and potential execution on Day T.

After Market Close on Day T-1:
The system scans the daily charts for the core pullback pattern and applies EOD quality filters to generate a "Watchlist" for the next day.

During the Trading Day on Day T:
The system monitors the 15-minute data for stocks on the Watchlist. When a stock breaks its trigger price, it runs the real-time "Advanced Conviction & Risk Engine" before executing a trade.

4.2. The HTF (Weekly) Strategy (htf_simulator_advanced.py)
This strategy uses a weekly cadence, separating setup discovery ("Scouting") from trade execution ("Sniping").

After Market Close on Friday (The "Scout" Mission):
The Scout scans the weekly charts for the core pullback pattern. For each valid setup, it generates a "Target List" for the entire upcoming week.

During the Following Week, Monday to Friday (The "Sniper" Mission):
The Sniper monitors the 15-minute data for stocks on the Target List. When a breakout occurs, it validates the move using the full Advanced Conviction & Risk Engine before entering.

4.3. The Monthly Strategy (simulator_monthly_advanced.py)
This strategy adapts the Scout/Sniper model for a much longer timeframe.

After Market Close on the Last Trading Day of the Month (The "Scout" Mission):
The Scout scans the monthly charts for the core pullback pattern. It applies quality filters and generates a "Target List" valid for the entire next month.

During the Following Month (The "Sniper" Mission):
The Sniper monitors the Target List intraday. It operates within an Adaptive Execution Window, typically ignoring the first few volatile days of the month and then monitoring for a set period for a breakout, which is then validated against the Conviction Engine.

Section 5: The Bias-Free Simulators & Core Engines
5.1. The Simulators
simulator_daily_hybrid.py: The advanced simulator for the daily strategy, now enhanced with flexible stop-loss modes and full MAE logging.

htf_simulator_advanced.py: The flagship simulator for the weekly HTF strategy. This script is now highly flexible, featuring toggleable modes for both the initial stop-loss (LOOKBACK vs. PERCENT) and the overall exit strategy (TRAILING vs. ATR_TARGET).

simulator_monthly_advanced.py: The new, state-of-the-art simulator for the monthly strategy, featuring unique adaptations like a configurable stop-loss mode (ATR vs. PERCENT) and a custom profit-taking mode.

5.2. The Advanced Conviction & Risk Engine
This engine is a core component of all advanced simulators. It is a suite of bias-free, intraday filters used to validate entry signals.

Time-Anchored Volume Projection: Projects final EOD volume based on cumulative intraday volume at specific time anchors.

VIX-Adaptive Market Strength & Slippage: Adjusts market health thresholds and slippage based on the previous day's VIX close.

Intraday Relative Strength (RS) Filter: Compares the stock's intraday performance against the index's performance at the moment of breakout.

Integrated Portfolio Risk Gate: Calculates position size based on both per-trade risk and total portfolio risk exposure, using the settled equity from the start of the day to prevent using unrealized profits as leverage. This is a critical calculation integrity feature.

Section 6: Strategic Roadmap & Current Status
6.1. Strategic Roadmap
Phase 0: Foundational Validation & Refinement (Complete)

Phase 1: Implement Core Risk Architecture (Complete)

Phase 2: Sharpen the Alpha & Build Conviction Engine (Complete)

Phase 3: Address Systemic Backtesting Flaws (In Progress)

Objective: Eliminate systemic biases to produce truly realistic performance metrics.

Phase 3.5: Advanced Strategy Analysis & Optimization (New)

Objective: Move beyond basic backtesting to perform a deep, professional-grade analysis of the finalized strategies to optimize risk management and understand their behavior in different market conditions. This will be guided by the Professional Strategy Analysis Checklist.

Phase 4: The Final Frontier (Predictive Modeling)

6.2. The Professional Strategy Analysis Checklist
This checklist is the official guide for Phase 3.5 of the project.

✅ 1. Excursion Analysis (MAE & MFE)

Objective: To analyze the price journey of each trade to optimize stop-loss and profit-taking rules.

Actionable Insights: Set a tighter atr_multiplier_stop based on MAE analysis of winners. Introduce a more conservative or dynamic profit target based on MFE analysis of losers.

✅ 2. Trade Duration Analysis

Objective: To determine if there is an optimal holding period for trades.

Actionable Insights: Implement time-based stops (e.g., "slow mover exit") if analysis reveals that trades held beyond a certain duration are more likely to fail.

✅ 3. Equity Curve & Drawdown Analysis

Objective: To understand the frequency, duration, and depth of drawdowns.

Actionable Insights: Use the analysis to refine risk parameters and for psychological preparedness for live trading.

✅ 4. Regime Analysis

Objective: To identify the specific market environments where the strategy thrives and where it struggles.

Actionable Insights: Implement a master "regime filter" to disable the strategy during unfavorable market conditions or dynamically adjust position sizing based on the regime.

✅ 5. Portfolio Construction & Correlation Analysis

Objective: To prevent large drawdowns caused by over-concentration in a single factor (e.g., a specific sector).

Actionable Insights: Implement portfolio-level risk controls such as:

Volatility-Adaptive Position Sizing: Risk less on highly volatile stocks and more on stable stocks.

Sector Exposure Caps: Set a hard limit on the capital allocated to any single market sector.

Correlation-Based Diversification: Reject new trades that are too highly correlated with existing open positions.

6.3. Current Project Status & Immediate Next Steps
Current Status: The project has successfully developed a suite of robust, bias-free simulators for daily, weekly, and monthly timeframes, complete with a standardized and detailed logging system. The calculation integrity of the position sizing logic has been corrected across all simulators. The immediate focus remains on addressing the foundational backtesting flaws before moving to the advanced analysis phase.

Immediate Next Steps:

Implement a Transaction Cost Model: Integrate a realistic cost model into all simulators to account for brokerage, STT, and other fees.

Mitigate Survivorship Bias: Research and implement a solution for using point-in-time historical index constituents instead of a static nifty200.csv file. This is the highest priority for achieving trustworthy backtest results.