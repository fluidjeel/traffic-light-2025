Master Project Context: Nifty 200 Pullback Strategy
Version: 3.2
Date: 2025-07-27
Status: Foundational Validation Complete. Ready for Phase 1.

1. High-Level Project Goal & Philosophy
1.1. Goal
The primary objective of this project is to develop, backtest, and ultimately automate a profitable, long-only, pullback-based swing trading strategy with a verifiable and realistic edge. The strategy is applied to the Nifty 200 stock universe. The key performance indicators are to achieve a high Compound Annual Growth Rate (CAGR) while maintaining acceptable portfolio drawdown levels.

1.2. Core Philosophy
Our guiding principle is an unwavering commitment to eliminating lookahead bias. All development and testing must rigorously distinguish between "Golden Benchmarks" (biased, theoretical models that represent the strategy's maximum potential) and "Realistic Simulators" (bias-free, actionable models that replicate real-world trading conditions).

2. System Architecture & Data Pipeline
The project is a modular pipeline designed for rigorous, bias-free research. Each script has a specific and specialized role, starting with data acquisition and processing.

2.1. Data Acquisition & Processing Workflow
Data Acquisition: Raw daily and 15-minute historical data for all stocks and indices in the Nifty 200 universe is fetched using a suite of fyers_scrapers.

Data Processing & Resampling: The calculate_indicators_clean.py script acts as the central data processing engine. It performs two critical functions:

Resampling: It takes the raw daily data and resamples it into all required higher timeframes (e.g., 'W-FRI' for weekly, 'MS' for monthly) by aggregating the open, high, low, close, and volume data.

Indicator Calculation: It then calculates a comprehensive suite of technical indicators (EMAs, SMAs, RS, ATR) for all timeframes and saves these clean, enriched data files to the data/processed directory. This ensures the backtesters have consistent and accurate data to work with.

3. Core Trading Strategy Logic
The strategy is designed to identify a short-term pullback within a prevailing uptrend and enter on a sign of trend resumption. The rules are applied consistently across both the Daily and Higher-Timeframe (HTF) versions of the strategy.

3.1. Setup Identification (The Pattern)
The core pattern is a sequence of one or more red candles followed by a green candle. This pattern is the foundational building block of a valid setup.

3.2. Setup Quality Filters
To ensure only high-probability patterns are considered, a series of strict quality filters are applied to the green setup candle:

Trend Filter: The green candle must close above its 30-period EMA.

Weakness Filter: The green candle must close in the lower half of its own range (close < (high + low) / 2). This is a critical alpha-generating filter designed to avoid chasing exhaustive moves and instead find setups that are pausing before a potential continuation.

3.3. Entry Conditions & Confirmation Filters
A setup is only considered valid for a trade if it meets the following conditions on the trigger day (the day after the setup candle):

Price Breakout: The high of the trigger day's candle must cross above the high of the setup pattern.

Market Regime Filter: The Nifty 200 Index must be trading above its 50-day EMA. New long trades are only initiated when the broader market is in a confirmed uptrend.

Volume Filter: The volume on the trigger day must be at least 1.3 times its 20-day average volume. This confirms institutional interest in the breakout.

Relative Strength (RS) Filter: The stock's 30-day price return must be greater than the Nifty 200's 30-day price return. The strategy aims to only trade market leaders.

3.4. Trade Management Rules
Once a position is opened, it is managed by a clear set of rules:

Initial Stop-Loss: The stop-loss is placed at the lowest low of the 5 candles preceding the entry.

Partial Profit Target: A 1:1 risk/reward target is calculated. If the price hits this target, half the position is sold, and the stop-loss for the remaining half is moved to the entry price (breakeven).

Trailing Stop-Loss: For the remaining half of the position, the stop-loss is trailed upwards. On any subsequent day that closes green, the stop is moved up to that day's low.

4. Execution Timeline & Logic Breakdown
This section details the precise timing and calculation methods for each component of the realistic simulators.

4.1. Daily Simulator (simulator_daily_hybrid.py)
Post-Market / End-of-Day (EOD) on Day T:

Action: Generate Watchlist for Day T+1.

Process: The script iterates through all stocks. For each stock, it checks the completed daily candle for Day T to see if it qualifies as a valid setup candle (green, preceded by red, above 30 EMA, closed in lower half). If the pattern is valid, it then applies the Volume and RS filters using the data from Day T. Stocks that pass all checks are added to the watchlist for Day T+1.

Action: Manage Open Positions.

Process: For all currently open trades, the script checks the completed daily candle for Day T and applies the trailing stop-loss logic, including the aggressive_breakeven rule if enabled.

Pre-Market on Day T+1:

Action: Gap-Up Filter.

Process: The script checks the opening price for all stocks on the watchlist. If a stock's open is significantly above its trigger price, it is removed from the watchlist for the day to avoid chasing an excessive gap.

Intraday during Day T+1:

Action: Execute Entries.

Process: The script monitors the 15-minute data for stocks remaining on the watchlist. When an intraday candle's high crosses the trigger price, it immediately checks the real-time conviction filters (Intraday Market Strength, Volume Velocity). If they pass, an entry order is simulated.

Action: Manage Open Positions.

Process: The script monitors the 15-minute data for all open positions. It checks for exits based on the partial profit target or the current stop-loss level.

4.2. HTF Simulator (simulator_htf_scout_sniper.py)
Post-Market / End-of-Day (EOD) on Day T:

Action: "Scout" runs to generate Target List for Day T+1.

Process: The Scout scans all stocks. It first identifies stocks with a valid weekly setup pattern (weekly green candle, preceded by red, etc.). It then checks the completed daily candle for Day T to see if a breakout above the weekly trigger price has already occurred. If the breakout is confirmed, it applies the daily Volume and RS filters. Stocks that pass all checks are added to the Target List for Day T+1.

Action: Manage Open Positions.

Process: Same as the daily simulator; trailing stop logic is applied based on the completed daily candle for Day T.

Pre-Market on Day T+1:

Action: imminence_filter (if enabled).

Process: Before the market opens, this filter can be applied to the Target List. It checks if the prior day's (Day T) candle was an "Inside Day" or "NR7" pattern. Only stocks showing this sign of recent volatility contraction are kept on the Target List for the Sniper to monitor.

Intraday during Day T+1:

Action: "Sniper" executes entries.

Process: The Sniper monitors the 15-minute data for stocks on the Target List only. It is not waiting for a price breakout. It is waiting for a 15-minute candle that shows sufficient conviction (passes Intraday Market Strength and Volume Velocity filters) to confirm the breakout from the prior day has momentum. If conviction is found, an entry order is simulated.

Action: Manage Open Positions.

Process: Same as the daily simulator.

5. Feature Calculation Logic
EMA (Exponential Moving Average): Calculated using a standard EMA formula, typically with a 30-period lookback on the relevant timeframe (daily or weekly). pandas_ta.ema(df['close'], length=30)

Volume SMA (Simple Moving Average): A 20-period simple moving average of the daily volume. pandas_ta.sma(df['volume'], length=20)

Relative Strength (RS): Calculated as the percentage price change over the last 30 periods. df['close'].pct_change(periods=30) * 100

Initial Stop-Loss: The lowest low value within the 5 daily candles that precede the entry candle.

Risk Per Share: Entry Price - Initial Stop-Loss

Partial Profit Target: Entry Price + Risk Per Share

Aggressive Breakeven: An EOD rule. If a position is profitable at the end of the day, the stop-loss for the next day is moved to Entry Price + breakeven_buffer_points. This is applied to protect small gains. It is applicable to both Daily and HTF simulators.

Imminence Filter: A pre-market filter that checks for volatility contraction on the prior day's candle.

Inside Day: today['high'] < yesterday['high'] and today['low'] > yesterday['low']

NR7 (Narrowest Range of 7 Days): today_range == last_7_days_range.min()

This filter is specific to the HTF simulator only.

Intraday Market Strength: A boolean check. NIFTY200_INDEX_15min_close > NIFTY200_INDEX_day_open

Volume Velocity: A boolean check. intraday_cumulative_volume > (average_daily_volume * threshold_pct)

6. Strategic Roadmap for Optimization
Phase 0: Foundational Validation & Refinement (Complete)
Objective: Solidify the integrity of the backtesting framework.

Tasks:

Complete subset validation for both daily and HTF strategies.

Finalize and lock the core logic of the benchmark and simulator scripts.

Conduct parameter sensitivity analysis to find the optimal baseline parameters.

Perform a final lookahead bias audit.

Phase 1: Implement Core Risk Architecture
Objective: Control portfolio-level risk.

Tasks:

Implement a hard limit on the number of concurrent open positions.

Implement a portfolio-wide daily drawdown limit (a "kill switch").

Phase 2: Sharpen the Alpha & Build Conviction Engine
Objective: Improve the quality of entry signals.

Tasks:

Implement an Adaptive Volume Velocity filter based on the VIX.

Implement an Intraday Relative Strength filter.

Develop a composite conviction score to rank setups.

Phase 3: Implement Dynamic Risk
Objective: Link capital allocation directly to signal quality.

Tasks:

Implement dynamic position sizing based on the conviction score.

Add advanced confirmation filters (e.g., Multi-Timeframe alignment).

Phase 4: The Final Frontier (Predictive Modeling)
Objective: Bridge the final performance gap using advanced techniques.

Tasks:

Develop a predictive ML filter to identify high-probability setups.

Develop a "Second Chance" protocol to re-enter missed breakouts.