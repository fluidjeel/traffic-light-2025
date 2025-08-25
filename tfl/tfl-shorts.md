Strategy Document: TrafficLight-Manny (Shorts)
1. Overall Strategy Overview
The TrafficLight-Manny (Shorts) is a quantitative, momentum-following strategy designed to systematically capture profits from strong downward price movements in the Indian F&O stock universe.

It operates on a 15-minute timeframe and uses a multi-layered filtering system to identify high-probability short-selling opportunities. The core principle is to short instruments that are already exhibiting significant weakness on both a daily and intraday basis, following a minor short-term pullback.

The strategy employs a sophisticated, multi-stage trade management system designed to protect capital and maximize profits by allowing "homerun" trades to run as far as possible.

2. Hypothesis
The core hypothesis of this strategy is that:

An instrument that is already in a confirmed downtrend (as measured by daily RSI and intraday MVWAP/EMA) and experiences a brief, multi-candle pullback (a "pause" in the downtrend) is highly likely to resume its downward trajectory.

By entering a short position as the instrument breaks below the low of this pullback pattern, we can capture the subsequent wave of selling pressure with a clearly defined and limited initial risk.

3. Entry and Exit Rules
Entry Signal
A short trade is triggered on a 15-minute candle when all of the following conditions are met:

Pattern Detection: Within the last 10 candles, there must be a sequence of at least 1 consecutive green candle followed immediately by a red "signal" candle.

Entry Trigger: The current candle's low must break below the lowest low of the identified pattern candles.

Confirmation Filters
The entry signal is only considered valid if it passes a series of confirmation filters:

Daily Momentum Filter: The daily RSI (14-period) of the instrument must be below 50, indicating it is already in a state of bearish momentum.

Intraday Trend Filter (Adaptive):

For stocks, the current 15-minute price must be trading below its 50-period Moving Volume Weighted Average Price (MVWAP).

For indices (which lack volume), the current 15-minute price must be trading below its 50-period Exponential Moving Average (EMA).

Initial Stop Loss & Take Profit
Initial Stop Loss: Placed at the highest high of the identified pattern candles.

Initial Take Profit: A distant target is set at 10R (10 times the initial risk). This target is designed to rarely be hit, allowing the trailing stop mechanisms to manage the trade exit.

4. Trade Management
The strategy employs a three-stage, dynamic trade management system:

Breakeven Stop: Once the trade moves in our favor and reaches a profit of 0.5R, the stop loss is immediately moved to lock in a small profit of 0.1R. This protects the trade from turning into a loser.

Standard Trailing Stop: A standard ATR-based trailing stop is active from the start of the trade, using a 2.0x multiplier on the 14-period ATR. This allows the trade room to develop.

Aggressive Trailing Stop: If the trade becomes a "homerun" and reaches a profit of 3.0R, the trailing stop automatically becomes more aggressive, tightening to a 1.0x multiplier on the 14-period ATR to lock in gains more quickly.

5. Code Explained and Folder Structure
The system is organized into a series of Python scripts and data folders.

Folder Structure
/algo-2025/
|-- /data/
|   |-- /universal_historical_data/  (Raw .csv data from scraper)
|   |-- /universal_processed/        (Indicator-rich .parquet files)
|       |-- /daily/
|       |-- /15min/
|-- /tfl/                           (Strategy-specific code)
|   |-- main_simulator.py
|   |-- shorts_simulator.py
|   |-- mae_mfe_analyzer.py
|-- /backtest_logs/                 (All results are saved here)
|   |-- /TrafficLight-Manny-SHORTS_ONLY/
|       |-- /20250825_210000/
|           |-- summary.txt
|           |-- trade_log.csv
|           |-- advanced_analysis.csv
|-- universal_fyers_scraper.py
|-- universal_calculate_indicators.py
|-- convert_to_parquet.py
|-- nifty200_fno.csv

Key Scripts
universal_fyers_scraper.py: Downloads raw 15-minute and daily price data.

universal_calculate_indicators.py: Calculates all necessary technical indicators (EMAs, RSI, MVWAP, etc.) on the raw data.

convert_to_parquet.py: Converts the processed CSV files into the much faster Parquet format.

shorts_simulator.py: The core backtesting engine for the short strategy. It reads the Parquet data, applies the rules, and generates performance logs.

mae_mfe_analyzer.py: A powerful analysis tool that reads the trade_log.csv to provide deep insights into trade performance (MFE, MAE, RSI analysis) to guide optimization.

6. Realism Checklist
To bridge the gap between backtest results and live trading performance, the following factors must be considered:

Commissions & Charges: The current backtest does not account for brokerage, taxes, or exchange fees. The 0.1R profit buffer on the breakeven stop is a partial mitigation for this.

Slippage: The backtest assumes perfect entry and exit fills. In live trading, slippage will cause slightly worse fills, which will impact profitability.

Data Quality: The system is highly dependent on clean, accurate historical data. Any errors in the data files will lead to unrealistic backtest results.

Lookahead Bias: The vectorized backtester has been carefully designed to prevent lookahead bias by pre-calculating signals and shifting them forward one candle.

7. Future Roadmap
The development of the TrafficLight-Manny strategy is an iterative process. The next logical steps are:

Optimize the Long Side: Create a longs_simulator.py and apply the same rigorous, data-driven analysis to develop a robust long strategy. This will likely involve defining a "Risk-On" market regime and identifying the optimal filters for long trades.

Merge into a Unified Strategy: Combine the optimized long and short logic into the main_simulator.py to create a single, all-weather trading system.

Capital Management: Implement a position sizing model (e.g., a percentage of equity per trade) to manage risk across the portfolio.

Walk-Forward Analysis: Perform a walk-forward optimization to validate the strategy's parameters and ensure they are robust across different time periods.