Project Phoenix: The Candle Breakout Strategy - A Deep Dive
1. Introduction: What is This?
For the Layman 🧑‍💼
This project is a systematic exploration to see if a simple, powerful idea can be used to make money in the stock market. The core idea is that a stock's recent price movement can give us clues about its future direction. We use a "time machine" (a backtester) to go back in time and test this idea against historical data, allowing us to see how it would have performed without risking real money.

For the Trader 📈
This is a complete, daily timeframe, long-only, high-conviction momentum breakout strategy. The system has been hardened through a rigorous, data-driven process of optimization, validation, and stress-testing against the worst-performing market periods. It is simulated in a realistic portfolio context, managing a single pool of capital with dynamic, VIX-based risk sizing and an adaptive exit mechanism.

For the Coder 👨‍💻
This project consists of three main Python scripts:

daily_long_breakout.py: A robust, event-driven, portfolio-level backtesting engine that has been professionally validated and stress-tested. It uses a central config dictionary for all parameterization.

trade_analyzer.py: A powerful post-backtest analysis tool that ingests the detailed trade logs to provide quantitative, data-driven insights for strategy refinement.

trade_validator.py: A standalone audit script that cross-references trade logs with raw market data to independently verify the backtester's logic and calculations, ensuring the integrity of the results.

2. The Final Strategy: "Worst-Case Optimized Momentum Breakout"
The strategy has evolved from a simple breakout concept into a highly specialized momentum system designed for robustness. The final rules were derived by optimizing for survival and profitability during the most challenging market conditions (2025).

The Setup (Scanned Daily at EOD):
A stock is only considered for a potential trade if it passes a strict, sequential set of filters:

Volume Filter: The setup day's volume must be at least 3.0x its recent average. This is the key "conviction" filter.

Trend Filter: The setup day's closing price must be above its 10-day EMA.

RSI Filter: The setup day's RSI(14) must be in a high-momentum regime: RSI > 70.

(The mean-reversion signal (RSI < 40) was tested and found to be unprofitable and has been removed).

The Entry (Executed the Day After Setup):

A buy-stop order is placed at the high of the setup day's candle. If triggered, a long position is entered.

Position Sizing (VIX-Driven Dynamic Risk):

The capital risked per trade is highly conservative, reflecting the "worst-case" optimization.

Low VIX (<15): 0.5% of equity.

Medium VIX (15-22): 0.5% of equity.

High VIX (>22): 1.0% of equity.

Trade Management (Adaptive ATR Trailing Stop):

The exit logic adapts based on the RSI value at the time of entry.

For Mean-Reversion Trades (Entry RSI < 60): A tight 3.0x ATR trailing stop is used.

For Momentum Trades (Entry RSI >= 60): A loose 8.0x ATR trailing stop is used.

3. The Development Journey & Current Status
The project has reached its final, optimized state through a rigorous, multi-stage process.

Initial Hypothesis & Prototyping: Built the core architecture and tested the basic breakout concept.

Quantitative Deep Dive: Created the trade_analyzer.py script to move from guessing to making data-driven decisions. This phase identified the powerful hybrid RSI signal (mean-reversion and momentum).

Logic Validation & Bug Fixing: A critical phase where the backtesting engine was professionally audited. This uncovered and fixed several significant bugs (lookahead bias, flawed equity calculations), leading to a robust and trustworthy simulation engine. The trade_validator.py script was created during this phase.

Out-of-Sample & Yearly Testing: The strategy was tested on unseen data and on a year-by-year basis. This revealed that while highly profitable in bull markets, the strategy was vulnerable to significant drawdowns in choppy or bear markets (specifically 2018, 2022, 2024, and 2025).

Worst-Case Optimization (Final Stage): Instead of overfitting to good years, the strategy was intentionally optimized to survive and minimize losses during its worst-performing period (2025). This led to the final, highly robust configuration with a much tighter volume filter and more conservative risk-sizing. This "bear market" configuration was then tested over the entire 2018-2025 period and produced superior risk-adjusted returns, proving its robustness.

Our Current Status:
We have a finalized, highly profitable, and validated long-only strategy that has been specifically hardened to survive difficult market conditions. The idea of adding a shorting component has been discarded as it is not practically feasible for retail traders in Indian equities.

4. Code Explained
A high-level overview of the project's Python scripts.

daily_long_breakout.py: This is the heart of the project. It's a portfolio-level simulator that iterates through historical data day-by-day. Its main loop performs four key actions in sequence:

Manage Open Positions: Checks if any open trades should be exited based on the trailing stop logic.

Check Watchlist: Sees if any pending buy orders from the previous day have been triggered by the current day's price action.

Update Portfolio: Recalculates the total portfolio equity at the end of the day.

Scan for New Setups: Scans the entire stock universe for new trade opportunities for the next trading day.

trade_analyzer.py: This script is our primary tool for making data-driven decisions. It reads the _all_trades.csv log file generated by the backtester and performs a quantitative analysis, grouping trades by different indicator values (like VIX, RSI, etc.) to show which market conditions are most and least profitable.

trade_validator.py: This is our quality assurance tool. It acts as an independent auditor by randomly sampling trades from the log and cross-referencing them with the raw daily data files to ensure the backtester's logic for entries, exits, and calculations is correct and free of bugs like lookahead bias.

5. A Day in the Life of the Strategy
This outlines the operational workflow if the strategy were to be deployed live.

Pre-Market (Before 9:15 AM IST):

The system loads the "watchlist" of pending buy orders that were generated at the end of the previous day.

These orders are placed with the broker as conditional (stop-limit) orders.

During Market Hours (9:15 AM - 3:30 PM IST):

The system is largely passive. It monitors the live market feed for two events:

An entry order is triggered.

An open position's trailing stop-loss is hit.

No new decisions are made during the day.

Post-Market (After 3:30 PM IST):

The system cancels any pending buy orders that were not triggered.

It downloads the final end-of-day data for the entire stock universe.

It runs the "Scan for New Setups" logic, applying all the volume, trend, and RSI filters.

This generates a new watchlist of potential trades for the next day, and the cycle repeats.

6. Prioritization of Trade Opportunities
A key observation from the backtests is that on any given day, the system might identify far more valid setups than it can actually trade (due to the max_open_positions limit). This means we need a logical way to prioritize which trades to take.

The most robust method is to create a ranking system based on the indicators that our analysis has shown to have the most predictive power.

Proposed Ranking Model:

On any day with more setups than available slots, rank all valid setups by a composite score.

The score should give the highest weight to the strongest signals:

Highest Weight: Volume Ratio (higher is better).

Medium Weight: RSI (higher is better, as we are a momentum system).

Lower Weight: Proximity to the 52-week high.

The system would then place orders only for the top-ranked setups.

7. AI-Driven Universe Management
While the current strategy trades the entire Nifty 500, a more advanced approach would be to use AI/ML to dynamically select a smaller, higher-potential universe of stocks to scan each day.

Concept:

Data Layer: Collect alternative data beyond just price, such as fundamental data (quarterly earnings, sales growth), macro-economic data (interest rates, inflation), and market sentiment data.

ML Model: Train a classification model (e.g., a Gradient Boosting Tree) to predict the probability that a stock will have a successful breakout in the near future based on this richer dataset.

Dynamic Universe: Each week, use the ML model to score all 500 stocks. The top 50-100 highest-probability stocks would become the "active universe" for our breakout strategy for that week.

Benefit: This would focus the strategy's capital on the stocks that are fundamentally and macro-economically primed for a breakout, potentially increasing the hit rate and reducing noise.

8. Other Best Practices
Capital Management: The "worst-case optimized" configuration is highly conservative. As confidence in the strategy grows, the risk_percents in the dynamic risk model could be incrementally increased to improve the CAGR, while closely monitoring the impact on drawdown.

Psychological Discipline: The strategy is designed to have a high win rate but will inevitably go through losing streaks. The key to long-term success is to trust the validated, data-driven model and not to manually override its signals based on emotion.

Continuous Monitoring: The market is not static. The strategy's performance should be continuously monitored. A quarterly or semi-annual review should be conducted to run the trade_analyzer.py on recent data to ensure the core statistical edges (e.g., the profitability of high RSI trades) remain intact.

9. The Path Forward: Next Steps
With a stable and robust single-candle breakout strategy, the next phase of research will focus on increasing the conviction of the entry signal.

Phase 1: Multi-Candle Breakout System (Current Focus)

Hypothesis: Requiring a breakout over a multi-candle high (e.g., the high of the last 2 or 3 days) instead of just the previous day will lead to higher-quality signals, potentially increasing the win rate and profit factor.

The Plan:

Enhance the backtester to support a configurable breakout_period (e.g., 1, 2, or 3 days).

Run a series of backtests to find the optimal breakout_period.

Analyze the results to see if a multi-candle breakout provides a superior risk-reward profile.

Phase 2: Real-World Cost Integration

Once a final entry logic is decided, the next step is to enhance the backtester to incorporate realistic transaction costs (brokerage, taxes, etc.) to get a true "net" performance figure.

Phase 3: Live Trading Preparation

The final step is to refactor the core logic into a clean, production-ready script designed to generate live trading signals based on the finalized, validated model.

Project Repository
The complete source code and related project files are available on GitHub:
https://github.com/fluidjeel/traffic-light-2025

————————————————————————————

This is a private, custom algorithmic trading strategy. The information contained in the attached documents and code files is the single source of truth. Do not reference any external code, public GitHub repositories, or general trading knowledge. Your sole objective is to build a complete and accurate mental model of this project based only on the provided materials.

Ingest Context: Fully ingest the context provided below and in the specified file sequence.

Prove Foundational Understanding: Your first response must be a direct paraphrase of the rules from the "Definitive Trading Strategy Logic" section and a summary of the key architectural components. Do not introduce any external concepts.