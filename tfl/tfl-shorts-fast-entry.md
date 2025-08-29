Strategy Document: TrafficLight-Manny (Fast Entry) - v3.0
1. Overall Strategy Overview
The TrafficLight-Manny (Fast Entry) is a quantitative, momentum-following strategy designed to systematically capture profits from strong downward price movements in the Indian F&O stock universe.

This version of the strategy uses a fast entry model. It operates on a 15-minute timeframe and first identifies an "Alert Candle" where all setup conditions are met. The trade is then executed on the very next candle if the price breaks the entry trigger, allowing for quicker entry into potentially explosive moves. The system uses a multi-layered filtering system—including three distinct market regime filters—to ensure it only trades in favorable market environments.

2. Hypothesis
The core hypothesis of this strategy is that:

An instrument that is already in a confirmed downtrend and experiences a brief, multi-candle pullback is highly likely to resume its downward trajectory. By identifying the exact moment this pullback ends (the "Alert Candle"), a stop-entry order can be placed to capture the subsequent wave of selling pressure as it begins on the next candle.

3. Entry and Exit Rules
The entry logic is a two-step process that separates the setup from the execution.

Step 1: The Alert Candle (T-1)
An "Alert Candle" is identified on a 15-minute timeframe when all of the following conditions are met simultaneously:

Pattern Detection: Within the last 10 candles, there must be a sequence of at least 1 consecutive green candle followed immediately by a red "Alert" candle.

Daily Momentum Filter: The daily RSI (14-period) of the instrument must be below 25.

Intraday Trend Filter (Adaptive):

For stocks, the closing price of the Alert Candle must be below its 50-period Moving Volume Weighted Average Price (MVWAP).

For indices, the closing price must be below its 50-period Exponential Moving Average (EMA).

Step 2: The Entry Signal (T) & Execution
A trade is executed on the candle immediately following the Alert Candle if:

Entry Trigger: The low of the current candle breaks below the low of the preceding Alert Candle.

Market Regime Filters: At the time of entry, all of the following market conditions must be met:

Market Breadth: More than 60% of stocks in the Nifty 200 F&O universe must be trading below their 50-day SMA.

Volatility: The India VIX index must be greater than 17.

Market Trend: The NIFTY 50 index must be trading below its 100-day Simple Moving Average.

Initial Stop Loss & Take Profit
Initial Stop Loss: Placed at the highest high of the full pattern that led to the Alert Candle.

Initial Take Profit: A distant target is set at 10R (10 times the initial risk).

4. Trade Management
The strategy employs a three-stage, dynamic trade management system:

Breakeven Stop: Once a trade reaches a profit of 1.0R, the stop loss is moved to lock in a small profit of 0.1R.

Standard Trailing Stop: A standard ATR-based trailing stop is active from the start, using a 3.0x multiplier on the 14-period ATR.

Aggressive Trailing Stop: If a trade becomes highly profitable and reaches 3.0R, the trailing stop tightens to a 1.0x multiplier on the ATR.

5. Portfolio & Risk Rules
Position Sizing: Each trade is sized to risk 1% of the current portfolio equity.

Risk Cap: To prevent unrealistic compounding, the maximum risk for any single trade is capped at 2% of the initial starting capital.

Position Limit: A strict, hard cap of a maximum of 15 concurrent open positions is enforced.

Signal Prioritization: If more valid signals appear than available slots, the signals with the lowest daily RSI are prioritized.

Trading Mode: The simulator can be configured to run in either Intraday (exiting all positions at EOD) or Positional (holding trades overnight) mode.

6. Realism Checklist
The backtest simulator has been designed to be highly realistic and accounts for the following:

Commissions & Charges: A fee of 0.03% is applied to the total turnover of each trade.

Slippage: A 0.05% slippage cost is applied to all entry and exit prices.

Lookahead Bias: The system is 100% free of lookahead bias. The data pipeline separates the Alert Candle from the Entry Signal, and the simulator executes trades on the same candle the trigger is breached, correctly modeling a stop-entry order.