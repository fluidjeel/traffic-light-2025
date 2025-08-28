Strategy Document: TrafficLight-Manny (Shorts) - v2.0
1. Overall Strategy Overview
The TrafficLight-Manny (Shorts) is a quantitative, momentum-following strategy designed to systematically capture profits from strong downward price movements in the Indian F&O stock universe.

It operates on a 15-minute timeframe and uses a multi-layered filtering system—including three distinct market regime filters—to identify high-probability short-selling opportunities. The core principle is to short instruments that are already exhibiting significant weakness on both a daily and intraday basis, following a minor short-term pullback, but only when the broader market environment is conducive to shorting.

The strategy employs a sophisticated, multi-stage trade management system designed to protect capital and maximize profits by allowing "homerun" trades to run as far as possible.

2. Hypothesis
The core hypothesis of this strategy is that:

An instrument that is already in a confirmed downtrend (as measured by daily RSI and intraday MVWAP/EMA) and experiences a brief, multi-candle pullback is highly likely to resume its downward trajectory, especially when the overall market is also showing signs of weakness or fear.

By entering a short position as the instrument breaks below the low of this pullback pattern, we can capture the subsequent wave of selling pressure with a clearly defined and limited initial risk.

3. Entry and Exit Rules
Entry Signal
A short trade is triggered on a 15-minute candle when all of the following conditions are met:

Pattern Detection: Within the last 10 candles, there must be a sequence of at least 1 consecutive green candle followed immediately by a red "signal" candle.

Entry Trigger: The current candle's low must break below the lowest low of the identified pattern candles.

Confirmation & Market Regime Filters
The entry signal is only considered valid if it passes a series of confirmation and market regime filters:

Daily Momentum Filter: The daily RSI (14-period) of the instrument must be below 25, indicating it is already in a state of strong bearish momentum.

Intraday Trend Filter (Adaptive):

For stocks, the current 15-minute price must be trading below its 50-period Moving Volume Weighted Average Price (MVWAP).

For indices, the current 15-minute price must be trading below its 50-period Exponential Moving Average (EMA).

Market Breadth Filter: The percentage of stocks in the Nifty 200 F&O universe trading below their 50-day SMA must be greater than 60%.

Volatility Filter: The India VIX index must be greater than 17, indicating a heightened level of market fear.

Market Trend Filter: The NIFTY 50 index must be trading below its 100-day Simple Moving Average.

Initial Stop Loss & Take Profit
Initial Stop Loss: Placed at the highest high of the identified pattern candles.

Initial Take Profit: A distant target is set at 10R (10 times the initial risk).

4. Trade Management
The strategy employs a three-stage, dynamic trade management system:

Breakeven Stop: Once the trade moves in our favor and reaches a profit of 1.0R, the stop loss is immediately moved to lock in a small profit of 0.1R.

Standard Trailing Stop: A standard ATR-based trailing stop is active from the start of the trade, using a 3.0x multiplier on the 14-period ATR.

Aggressive Trailing Stop: If the trade becomes a "homerun" and reaches a profit of 3.0R, the trailing stop automatically becomes more aggressive, tightening to a 1.0x multiplier on the 14-period ATR to lock in gains more quickly.

5. Portfolio & Risk Rules
Position Sizing: Each trade is sized to risk exactly 1% of the current portfolio equity.

Position Limit: A strict, hard cap of a maximum of 15 concurrent open positions is enforced.

Signal Prioritization: If more valid signals appear than available slots, the signals with the lowest daily RSI are prioritized.

6. Realism Checklist
The backtest simulator has been designed to be highly realistic and accounts for the following:

Commissions & Charges: A fee of 0.03% is applied to the total turnover of each trade to model brokerage, STT, and other exchange fees.

Slippage: A 0.05% slippage cost is applied to all entry and exit prices to model imperfect fills.

Lookahead Bias: The system is 100% free of lookahead bias. It uses a candle-by-candle, event-driven architecture that separates signal detection from trade execution, ensuring decisions are only made on data that was available at the time.