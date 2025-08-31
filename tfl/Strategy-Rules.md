TFL Automated Trading System: Official Strategy Rules
Document Version: 1.0
Applicable System Version: v3.0 and later

1. Introduction & Core Philosophy
This document provides a complete and verbose specification for the "TrafficLight-Manny" quantitative trading strategy. It is the single source of truth for all rules, parameters, and logic governing the system's decision-making process.

The core philosophy of the strategy is trend continuation with multi-layered confirmation. The system hypothesizes that an instrument already in a strong, confirmed trend (on both daily and intraday timeframes) which experiences a brief, minor pullback, is highly likely to resume its original trend.

The strategy only attempts to capture these trend-resumption moves when the broader market environment, or "regime," is conducive to either bullish (long) or bearish (short) price action. It is designed to be highly selective, prioritizing the quality of signals over the quantity of trades.

The system operates on a 15-minute timeframe for signal generation and trade execution, while leveraging daily data for higher-level momentum and regime filtering.

2. Long Strategy Rules (Bullish Scenario)
The system will only consider long trades if the daily market regime is permissive for bullish positions.

2.1. Market Regime Filters (The "Traffic Light" for Longs)
Before the market opens, the system analyzes the state of the broader market. All of the following conditions, based on the previous day's closing data, must be met for the system to even consider looking for long signals on a given day. If any condition fails, no long trades will be initiated.

Market Trend Condition: The overall market index must be in a confirmed uptrend.

Rule: The closing price of the NIFTY 50 Index must be greater than its 30-day Simple Moving Average (SMA).

Market Breadth Condition: The majority of stocks must be showing underlying strength, confirming broad market participation in the uptrend.

Rule: More than 60% of the stocks in the F&O universe must be trading above their respective 50-day SMAs.

Volatility Condition: This filter is currently disabled for the long strategy to capture moves in various volatility environments.

Rule: USE_VOLATILITY_FILTER is set to False.

2.2. Entry Signal Logic (The Setup)
If the market regime is permissive for longs, the system then scans all instruments in the universe on a 15-minute timeframe for the following precise setup. All conditions must be met on the same "signal candle" for a setup to be considered valid.

Price Action Pattern: A clear pattern of selling exhaustion followed by a reversal of momentum must occur.

Rule: A sequence of 1 to 9 consecutive red candles must be immediately followed by a single green "signal" candle.

Daily Momentum Filter: The instrument itself must be in a state of strong bullish momentum on a higher timeframe. This filter ensures the system is buying strength, not trying to pick bottoms.

Rule: The 14-period Daily Relative Strength Index (RSI) of the instrument must be greater than 75.

Intraday Trend Filter (Adaptive): The instrument must be in a confirmed uptrend on the current 15-minute timeframe.

Rule for Stocks: The current price must be above its 50-period Moving Volume Weighted Average Price (MVWAP).

Rule for Indices: The current price must be above its 50-period Exponential Moving Average (EMA).

2.3. Trade Execution (The Trigger)
Once a valid setup is identified on a signal candle (Candle T-1), the system waits for a price-based trigger on the subsequent candle(s) to execute the trade.

Trigger Price: The entry is triggered by the highest high of the entire red-and-green candle pattern.

Fast Entry Model: The system enters a trade on the very next candle (Candle T) if its price breaks above the trigger price. This is the default and tested mode.

Confirmed Entry Model: The system waits for a candle to close above the trigger price (Candle T) and then enters on the open of the next candle (T+1). This is an alternative, more conservative model.

2.4. Portfolio & Risk Management
The system employs a sophisticated, dynamic, multi-layered risk management framework to control exposure and size positions.

Position Sizing (Smart Risk Allocation):

The system first calculates a target risk amount for a new trade, which is 1% of the current total portfolio equity.

It then checks if adding this new risk would breach the portfolio's total heat limit. If Total Risk of Open Positions + New Trade's Risk > 5% of Equity, the system will attempt to resize the new trade to precisely fit the remaining available risk budget. The trade is only rejected if the remaining budget is too small for a viable position.

A final check ensures the total capital deployed for the trade does not exceed 25% of the current portfolio equity, preventing over-concentration in a single position. The final position size is the minimum of the quantity calculated by risk and the quantity calculated by this capital cap.

Portfolio Constraints:

A hard limit of 10 concurrent open positions is enforced.

If more valid signals appear than available slots, the system prioritizes the signals with the highest Daily RSI, ensuring it always attempts to enter the strongest momentum trades first.

2.5. In-Trade Management (The Exits)
Once a long trade is open, it is managed by a precise, multi-stage exit logic system.

Initial Stop-Loss (SL): The initial SL is placed at the lowest low of the identified red-and-green candle pattern.

Initial Take-Profit (TP): The initial TP is set at a distance equal to 10 times the initial risk (10R).

Breakeven Stop: Once the trade moves favorably and reaches a profit of 1.0R (one times the initial risk), the stop-loss is moved to lock in a small profit of 0.1R.

Multi-Stage ATR Trailing Stop: A dynamic trailing stop is active throughout the life of the trade.

Standard Mode: The trailing stop is placed at a distance of 4.0 times the 14-period ATR below the candle's high.

Aggressive Mode: If the trade becomes a "home run" and reaches a profit of 5.0R, the trailing stop automatically tightens to a more aggressive 1.0 times the 14-period ATR.

End-of-Day Exit: If EXIT_ON_EOD is True, any open position is automatically closed at the 15:15 IST candle.

3. Short Strategy Rules (Bearish Scenario)
The system will only consider short trades if the daily market regime is permissive for bearish positions. The rules are the symmetrical opposite of the long strategy.

3.1. Market Regime Filters (The "Traffic Light" for Shorts)
All of the following conditions, based on the previous day's closing data, must be met for the system to consider looking for short signals.

Market Trend Condition: The overall market index must be in a confirmed downtrend.

Rule: The closing price of the NIFTY 50 Index must be less than its 100-day Simple Moving Average (SMA).

Market Breadth Condition: The majority of stocks must be showing underlying weakness, confirming broad market participation in the downtrend.

Rule: More than 60% of the stocks in the F&O universe must be trading below their respective 50-day SMAs.

Volatility Condition: The market must be in a state of heightened fear or uncertainty, which is typically favorable for shorting.

Rule: The closing value of the INDIA VIX Index must be greater than 17.

3.2. Entry Signal Logic (The Setup)
If the market regime is permissive for shorts, the system scans all instruments on a 15-minute timeframe for the following precise setup.

Price Action Pattern: A clear pattern of buying exhaustion followed by a reversal of momentum must occur.

Rule: A sequence of 1 to 9 consecutive green candles must be immediately followed by a single red "signal" candle.

Daily Momentum Filter: The instrument itself must be in a state of strong bearish momentum on a higher timeframe. This ensures the system is shorting weakness.

Rule: The 14-period Daily Relative Strength Index (RSI) of the instrument must be less than 40.

Intraday Trend Filter (Adaptive): The instrument must be in a confirmed downtrend on the current 15-minute timeframe.

Rule for Stocks: The current price must be below its 50-period Moving Volume Weighted Average Price (MVWAP).

Rule for Indices: The current price must be below its 50-period Exponential Moving Average (EMA).

3.3. Trade Execution (The Trigger)
Once a valid setup is identified, the system waits for a price-based trigger.

Trigger Price: The entry is triggered by the lowest low of the entire green-and-red candle pattern.

Fast Entry Model: The system enters a trade on the very next candle if its price breaks below the trigger price.

3.4. Portfolio & Risk Management
The risk framework is symmetrical to the long side, with one key difference in prioritization.

Position Sizing: The same dynamic, multi-layered risk logic is applied (1% equity risk, 5% total portfolio risk, 25% capital per trade).

Portfolio Constraints:

A hard limit of 10 concurrent open positions is enforced.

If more valid signals appear than available slots, the system prioritizes the signals with the lowest Daily RSI, ensuring it always attempts to enter the weakest momentum trades first.

3.5. In-Trade Management (The Exits)
Once a short trade is open, it is managed by a symmetrical, multi-stage exit logic.

Initial Stop-Loss (SL): The initial SL is placed at the highest high of the identified green-and-red candle pattern.

Initial Take-Profit (TP): The initial TP is set at a distance equal to 10 times the initial risk (10R).

Breakeven Stop: Once the trade moves favorably and reaches a profit of 1.0R, the stop-loss is moved to lock in a small profit of 0.1R.

Multi-Stage ATR Trailing Stop: A dynamic trailing stop is active.

Standard Mode: The trailing stop is placed at a distance of 3.0 times the 14-period ATR above the candle's low.

Aggressive Mode: If the trade becomes a "home run" and reaches a profit of 3.0R, the trailing stop tightens to 1.0 times the 14-period ATR.

End-of-Day Exit: If EXIT_ON_EOD is True, any open position is automatically closed at the 15:15 IST candle.