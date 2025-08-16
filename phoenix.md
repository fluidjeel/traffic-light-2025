Project Phoenix: The Adaptive Breakout Strategy - A Deep Dive
1. Introduction: What is This?
For the Layman 🧑‍💼
This project is a systematic exploration to see if a simple, powerful idea can be used to make money in the stock market. The core idea is that a stock's recent price movement can give us clues about its future direction. We use a "time machine" (a backtester) to go back in time and test this idea against historical data, allowing us to see how it would have performed without risking real money.

For the Trader 📈
This is a complete, daily timeframe, long-only, high-conviction momentum breakout strategy. The system is designed for robustness by dynamically adapting its behavior to the current market environment. It uses a dual moving average filter on the NIFTY 500 to switch between an "Offensive Playbook" in strong uptrends and a "Defensive Playbook" in weakening or choppy markets. This allows the strategy to maximize gains in favorable conditions while preserving capital during unfavorable ones.

For the Coder 👨‍💻
This project consists of three main Python scripts:

daily_long_breakout.py: A robust, event-driven, portfolio-level backtesting engine. It features an adaptive logic module that dynamically adjusts key trading parameters (risk, stop-loss multipliers, entry filters) based on a configurable market regime filter. It includes a realistic model for Indian equity transaction costs.

trade_analyzer.py: A powerful post-backtest analysis tool that ingests the detailed trade logs to provide quantitative, data-driven insights for strategy refinement.

universal_calculate_indicators.py: The foundational data processing script that calculates all necessary technical indicators from raw price data, preparing it for the backtester.

2. The Final Strategy: "Adaptive Regime Breakout"
The strategy has evolved from a static ruleset into a dynamic system that changes its personality based on the health of the broad market. It uses a 30/50 day SMA filter on the NIFTY500 index to determine the market regime.

Market Regimes
Strong Uptrend (Offensive Playbook)

Condition: NIFTY 500 close is > 30 SMA and > 50 SMA.

Goal: Maximize gains from strong momentum. The system uses its most aggressive set of parameters.

Weakening (Defensive Playbook)

Condition: NIFTY 500 close is < 30 SMA but > 50 SMA.

Goal: Preserve capital. The system recognizes short-term weakness and switches to a highly conservative set of parameters.

Downtrend (Risk-Off)

Condition: NIFTY 500 close is < 50 SMA.

Action: The system stops looking for new trades entirely and only manages existing positions.

The Ruleset (Scanned Daily at EOD)
The Setup (Shared Logic):
A stock is only considered for a potential trade if it passes a strict, sequential set of filters:

Volume Filter: The setup day's volume must be at least 3.0x its 20-day average.

Trend Filter: The setup day's closing price must be above its 10-day EMA.

RSI Filter: The setup day's RSI(14) must be in a valid momentum or mean-reversion range (RSI < 40 or RSI > 70).

The Entry (Multi-Candle Breakout):

A buy-stop order is placed at the highest high of the last 2 days (configurable).

The initial stop-loss is placed at the lowest low of the last 2 days.

Trade Management (Intelligent Breakeven):

The strategy uses an intelligent, cost-aware breakeven mechanism.

The stop-loss is only moved up if the EOD closing price is high enough to cover all transaction costs (entry + estimated exit), ensuring a "scratch" trade is truly a net-zero PnL event.

The Adaptive Parameters
This is the core of the strategy. The following parameters change automatically based on the market regime:

Parameter

Offensive Playbook (Strong Uptrend)

Defensive Playbook (Weakening)

Max Distance from EMA

100% (No limit)

10% (Filters over-extended trades)

Risk (Medium VIX)

0.5% of equity

0.25% of equity (Cuts risk in half)

Momentum Trail Stop

8.0x ATR (Loose)

4.0x ATR (Tight)

Mean-Reversion Trail Stop

3.0x ATR (Standard)

2.0x ATR (Tight)

3. The Development Journey & Current Status
The project has reached its current state through a rigorous, multi-stage process that prioritized robustness and calculation integrity.

Initial Hypothesis & Prototyping: Built the core architecture and tested the basic single-candle breakout concept.

Quantitative Deep Dive: Created trade_analyzer.py to make data-driven decisions, identifying the hybrid RSI signal.

Logic Validation & Lookahead Bias Fix: A critical phase where the backtesting engine was audited. This uncovered and fixed a significant lookahead bias in the trailing stop calculation.

Worst-Case Optimization: The strategy was intentionally optimized to survive its worst-performing historical period (2022). This led to the creation of the "Defensive Playbook" parameters.

Calculation Integrity Audit: A deeper audit fixed several subtle but critical calculation flaws related to risk sizing, gap-down exits, and high/low watermark tracking.

Adaptive Regime Implementation: Instead of being permanently defensive, the system was upgraded to be adaptive. It now uses the "Offensive Playbook" during strong markets and automatically switches to the hardened "Defensive Playbook" when the market shows weakness.

Cost-Aware Realism (Final Stage): A realistic model for Indian equity transaction costs was integrated, and the breakeven logic was enhanced to be fully cost-aware.

Our Current Status:
We have a finalized, validated, and robust long-only strategy that adapts its behavior to the market. It operates with a realistic cost model and aims for both profitability in bull runs and capital preservation in choppy conditions.

4. Code Explained
daily_long_breakout.py: This is the heart of the project. It's a portfolio-level simulator that iterates through historical data day-by-day. Its main loop performs five key actions in sequence:

Determine Market Regime: Checks the NIFTY 500 index to select the Offensive or Defensive playbook for the day.

Manage Open Positions: Calculates and checks trailing stops (ATR and intelligent breakeven) using T-1 data to avoid lookahead bias.

Check Watchlist: Sees if any pending multi-candle breakout orders have been triggered.

Update Portfolio: Recalculates the total portfolio equity at the end of the day.

Scan for New Setups: Scans the stock universe for new trade opportunities using the day's active parameter set.

trade_analyzer.py: Our primary tool for making data-driven decisions. It reads the _all_trades.csv log file and groups trades by different indicator values to show which market conditions are most and least profitable.

universal_calculate_indicators.py: The script that pre-processes raw price data, calculating all the technical indicators (SMAs, EMAs, RSI, ATR, etc.) needed by the backtester.

5. The Path Forward: Future Areas of Improvement
Based on the v31 backtest results, the next steps should focus on improving consistency and reducing drawdowns without sacrificing the core trend-following edge. The priorities are:

1. Highest Priority: Enhance the Regime Filter with a Volatility Layer
Problem: The data shows the Medium VIX (15-22) environment is a consistent money-loser, regardless of the market trend.

Hypothesis: By becoming even more defensive when VIX is in this "trap zone," we can significantly reduce losses during choppy years.

Action: Enhance the logic so the system uses the Defensive Playbook if the market is in a 'Weakening' trend OR if the VIX is in the 'Medium' range.

2. Medium Priority: Develop the Mean-Reversion Sub-Strategy
Problem: The mean-reversion signals (RSI < 40) are highly profitable with a 45% win rate, but the system rarely takes them (only 11 trades in 7.5 years).

Hypothesis: The current entry filters are too restrictive for mean-reversion setups.

Action: Test a logical hypothesis for a sub-strategy, such as: "If the setup RSI is < 40, then lower the min_volume_ratio requirement from 3.0 to 1.5."

3. Lower Priority: Introduce Partial Profit-Taking
Problem: The high payoff ratio (14.12) means the strategy often gives back a significant amount of open profit before the trailing stop is hit.

Hypothesis: Booking partial profits will increase the number of winning trades and reduce drawdowns, creating a smoother equity curve.

Action: Implement a robust rule, such as: "When a trade's gross profit reaches 3R (three times initial risk), sell 30-50% of the position and move the stop-loss for the remainder to the cost-adjusted breakeven price."

For the LLM:

Project Repository
The complete source code and related project files are available on GitHub:
https://github.com/fluidjeel/traffic-light-2025

————————————————————————————

This is a private, custom algorithmic trading strategy. The information contained in the attached documents and code files is the single source of truth. Do not reference any external code, public GitHub repositories, or general trading knowledge. Your sole objective is to build a complete and accurate mental model of this project based only on the provided materials.

Ingest Context: Fully ingest the context provided below and in the specified file sequence.

Prove Foundational Understanding: Your first response must be a direct paraphrase of the rules from the "Definitive Trading Strategy Logic" section and a summary of the key architectural components. Do not introduce any external concepts.