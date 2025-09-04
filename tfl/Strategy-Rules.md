TFL Automated Trading System: Official Strategy Rules
Document Version: 2.0
Applicable System Version: Unified Portfolio Simulator v2.9+

1. Introduction & Core Philosophy
This document provides a complete and verbose specification for the "TrafficLight-Manny" (TFL) quantitative trading strategy. It is the single source of truth for all rules, parameters, and logic governing the system's decision-making process.

The core philosophy of the strategy is trend-agnostic momentum capture. The system hypothesizes that an instrument exhibiting extreme momentum (on a daily timeframe) and a specific intraday pullback pattern is highly likely to experience a sharp, directional move, regardless of the broader market trend.

The system operates in a "Trend Agnostic" mode by default. On any given day, it actively scans for both long and short opportunities across the entire instrument universe and deploys capital to the highest-quality signals based on a sophisticated, multi-layered risk management framework.

The strategy operates on a 15-minute timeframe for signal generation and trade execution, while leveraging daily data for higher-level momentum filtering.

2. Long Strategy Rules (Bullish Scenario)
The system will consider long trades on any instrument that meets the following criteria.

2.1. The Price Action Pattern
The core of the strategy is a precise, multi-candle price action pattern that identifies a minor pullback followed by a resumption of upward momentum. The pattern is defined by a sequence of candles on a 15-minute chart, labeled for clarity as T-N through T.

Candles T-N to T-2 (The Pullback): This must be a continuous series of one or more red candles (where the close is less than the open). The system will correctly identify any continuous series of red candles, with a minimum of 1 (MIN_RED_CANDLES) and a maximum of 9 (MAX_RED_CANDLES).

Candle T-1 (The Setup Candle): This must be a single green candle (where the close is greater than the open) that immediately follows the series of red candles.

Candle T (The Entry Candle): This is the candle that immediately follows the green T-1 setup candle.

2.2. Setup Candle (T-1) Filters
For a pattern to be considered a valid "setup," the green T-1 candle must meet all of the following conditions at its close:

Daily Momentum Filter: The 14-period Daily Relative Strength Index (RSI) of the instrument must be greater than 75.0.

Intraday Trend Filter: The closing price of the 15-minute T-1 candle must be above its 50-period Moving Volume Weighted Average Price (MVWAP) for stocks, or its 50-period Exponential Moving Average (EMA) for indices.

2.3. Entry Trigger on Candle T
A long trade is executed on the Entry Candle (T) if and only if its high price breaks above the true highest high of the entire preceding pattern.

Pattern High Calculation: The pattern_high is defined as the absolute highest high price recorded across the entire pattern, including the green T-1 candle and all of its preceding consecutive red candles (T-2 to T-N).

Entry Condition: high of Candle T >= pattern_high from the T-1 pattern.

3. Short Strategy Rules (Bearish Scenario)
The short strategy is a perfect, symmetrical mirror of the long strategy.

3.1. The Price Action Pattern
Candles T-N to T-2 (The Pullback): A continuous series of one or more green candles.

Candle T-1 (The Setup Candle): A single red candle that immediately follows the series of green candles.

Candle T (The Entry Candle): The candle that immediately follows the red T-1 setup candle.

3.2. Setup Candle (T-1) Filters
Daily Momentum Filter: The 14-period Daily RSI of the instrument must be less than 25.0.

Intraday Trend Filter: The closing price of the 15-minute T-1 candle must be below its 50-period MVWAP/EMA.

3.3. Entry Trigger on Candle T
A short trade is executed on the Entry Candle (T) if and only if its low price breaks below the true lowest low of the entire preceding pattern.

Pattern Low Calculation: The pattern_low is defined as the absolute lowest low price recorded across the entire pattern, including the red T-1 candle and all of its preceding consecutive green candles (T-2 to T-N).

Entry Condition: low of Candle T <= pattern_low from the T-1 pattern.

4. Portfolio & Risk Management Framework
The system employs a sophisticated, multi-layered, and dynamic risk management framework that is applied symmetrically to both long and short trades.

4.1. The Equity Cap (Realism Governor)
To ensure realistic backtest results and model the constraints of market liquidity, the system uses an Equity Cap for all risk calculations.

EQUITY_CAP_FOR_RISK_CALC_MULTIPLE = 15: The equity used for position sizing is capped at 15 times the INITIAL_CAPITAL. Any profits earned beyond this level are tracked, but do not contribute to increasing the size of new positions.

4.2. Position Sizing Logic
The quantity for each new trade is determined by a three-stage filtering process:

Portfolio Risk Budget: The system first calculates the total available risk budget for the entire portfolio.

MAX_TOTAL_RISK_PCT = 0.05: The sum of the initial risk of all open positions cannot exceed 5% of the (capped) portfolio equity.

The system calculates the available_risk_budget and uses this to cap the risk of any new trade.

Per-Trade Risk: The desired risk for the new trade is calculated.

RISK_PER_TRADE_PCT = 0.01: Each trade is sized to risk a maximum of 1% of the (capped) portfolio equity.

The final risk_amount for the trade is the lesser of the desired 1% risk and the available portfolio risk budget.

Capital Concentration Limit: Finally, the system checks for capital concentration.

MAX_CAPITAL_PER_TRADE_PCT = 0.25: No single position is allowed to consume more than 25% of the (capped) portfolio equity.

The final quantity is the lesser of the quantity calculated from the risk amount and the quantity calculated from the capital concentration limit.

4.3. Portfolio Constraints
STRICT_MAX_OPEN_POSITIONS = 15: A hard limit of a maximum of 15 concurrent open positions is enforced.

Signal Prioritization: If more valid signals appear than available slots, the system prioritizes:

Longs: The signals with the highest Daily RSI.

Shorts: The signals with the lowest Daily RSI.

5. In-Trade Management (The Exits)
Once a trade is open, it is managed by a symmetrical, multi-stage exit logic.

Initial Stop-Loss (SL):

Longs: The initial SL is placed at the pattern_low of the identified pattern.

Shorts: The initial SL is placed at the pattern_high of the identified pattern.

Initial Take-Profit (TP): The initial TP is set at a distance equal to 10 times the initial risk (10R).

Breakeven Stop: Once a trade moves favorably and reaches a profit of 1.0R, the stop-loss is moved to lock in a small profit of 0.1R.

Multi-Stage ATR Trailing Stop: A dynamic trailing stop is active.

Standard Mode:

Longs: Trails at a distance of 4.0 times the 14-period ATR below the candle's high.

Shorts: Trails at a distance of 3.0 times the 14-period ATR above the candle's low.

Aggressive Mode:

Longs: If a trade reaches a profit of 5.0R, the trailing stop tightens to 1.0 times the ATR.

Shorts: If a trade reaches a profit of 3.0R, the trailing stop tightens to 1.0 times the ATR.

6. Advanced Timing Rules
The system employs two final filters at the point of execution.

AVOID_OPEN_CLOSE_ENTRIES = True: The system will not initiate any new trades on the volatile 9:15 AM opening candle or the 3:15 PM pre-closing candle.

ALLOW_AFTERNOON_POSITIONAL = False: In the optimized configuration, this is set to False, meaning all trades are subject to the End-of-Day exit rule.

EXIT_ON_EOD = True: If a trade has not been closed by its SL, TP, or trailing stop, it will be force-closed at the market price of the 3:15 PM candle.