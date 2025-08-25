This is a private, custom algorithmic trading strategy. The information contained in the attached documents and code files is the single source of truth. Do not reference any external code, public GitHub repositories, or general trading knowledge. Your sole objective is to build a complete and accurate mental model of this project based only on the provided materials.

Ingest Context: Fully ingest the context provided below and in the specified file sequence.

Prove Foundational Understanding: Your first response must be a direct paraphrase of the rules from the "Definitive Trading Strategy Logic" section and a summary of the key architectural components. Do not introduce any external concepts.

Traffic Light 2025: The Daily Simulator - A Deep Dive
1. Introduction: What is This?
For the Layman 🧑‍💼
Imagine you have an idea for how to make money in the stock market. Instead of risking real money to see if it works, you could use a time machine. You'd go back in time, apply your idea to historical stock prices, and see what would have happened. This script is that time machine. It's a backtester, a program that simulates a trading strategy against past data to measure its performance. It tells you how profitable your idea might have been, how much risk it involved, and gives you a report card on its overall effectiveness.

For the Trader 📈
This is a complete, event-driven backtesting engine for a long-only, daily timeframe, pullback continuation strategy. It is designed to be as realistic as possible, simulating intraday entries and exits, slippage, and dynamic, portfolio-level risk management. Its purpose is to allow for rigorous, data-driven research and optimization of the "Traffic Light 2025" strategy by providing detailed performance metrics and trade logs.

For the Coder 👨‍💻
This is a monolithic Python script that uses the pandas library to perform a vectorized and event-driven backtest. It operates on pre-processed CSV files containing daily and 15-minute OHLCV data with pre-calculated technical indicators. The architecture is built around a central config dictionary for parameterization and a main run_backtest function that iterates through a master date index, simulating market behavior day by day to prevent lookahead bias. All portfolio management, state tracking, and logging are handled within this single, cohesive script.

Project Repository
The complete source code and related project files are available on GitHub:
https://github.com/fluidjeel/traffic-light-2025

2. The Core Strategy: "Pullback Continuation"
The fundamental idea behind this strategy is simple and powerful. We want to find stocks that are:

In a strong uptrend.

Have taken a short "breather" or pulled back for one or more days.

Are now showing signs of resuming that uptrend.

Our goal is to enter the trade at the exact moment the stock begins to move up again, catching the next wave of the trend.

3. The Control Panel: The config Dictionary
This is the heart of the simulator's flexibility. It's a single, large Python dictionary where every single rule, filter, and parameter of the strategy can be tuned without ever touching the core simulation code.

General & Data Settings
initial_capital: The starting amount of money for the simulation.

start_date / end_date: The time period for the backtest.

data_pipeline_config: Tells the script where to find the necessary data files (the processed daily/weekly data and the raw 15-minute intraday data).

Core Strategy Parameters
These settings define the basic rules for identifying a potential trade setup at the end of each trading day.

use_ema_filter: A master switch for the trend filter. If True, the strategy is active. If False, it will not look for trades.

ema_period: The primary tool for defining an uptrend. The script checks if a stock's closing price is above its 30-day Exponential Moving Average (EMA). This is a classic way to confirm the underlying trend is positive.

The Market Regime Filter: Our "Weather Report"
This is one of the most advanced features. It prevents the strategy from taking trades when the overall market is weak, even if an individual stock looks good.

For the Layman: Before trading, the script checks the "weather" of the overall stock market (represented by the NIFTY 200 index). If the market is stormy (in a downtrend), it wisely decides to stay on the sidelines to avoid unnecessary risk.

For the Trader: It acts as a top-down market filter.

market_regime_filter: The master switch for this entire system.

regime_index_symbol: The index used to gauge market health (NIFTY 200).

regime_ma_period: The primary check. If the NIFTY 200 is below its 50-day EMA, the market is considered in a "Bear Market," and no new trades are taken.

secondary_market_regime_filter: A more nuanced, secondary check. If the market is above its 50-day EMA but below its shorter-term 20-day EMA, it's considered to be "Weakening."

Dynamic Portfolio Risk: This system directly connects to the Market Regime.

In a "Healthy" market, it allows the portfolio to risk up to 10% of its total capital.

In a "Weakening" market, it immediately cuts the allowed risk in half to 5%, forcing a more defensive posture.

The "Elite" Entry Filters: Focusing on Quality
These are a series of strict rules designed to ensure we only trade the absolute best-looking setups.

filter_prox_52w_high: Only considers stocks that are within 10% of their 52-week high. This focuses the strategy on market leaders that are showing strong momentum.

filter_volume_ratio: Requires the setup day's volume to be at least 1.5 times its 20-day average. This is a confirmation that large institutions are interested in the stock.

filter_low_wick_candle: A price action filter. It rejects setups if the green candle has a large upper "wick" (greater than 10% of the candle's total range). A large wick is a sign of selling pressure, and this filter helps avoid trades that are likely to fail.

setup_candle_filter: Another price action filter. It ensures the setup candle is not an explicitly bearish pattern like a "Shooting Star," which also indicates selling pressure.

Trade Management: How We Handle a Live Trade
These rules govern what happens after we enter a position.

stop_loss_mode: Set to 'ATR' (Average True Range). This means the stop-loss is placed based on the stock's recent volatility, giving volatile stocks more room to move and keeping a tighter stop on less volatile ones.

atr_multiplier: The stop-loss is placed 2.5 times the ATR value below the entry price.

use_atr_trailing_stop: Once a trade is in profit, the stop-loss will automatically "trail" below the price, locking in profits as the stock moves up.

Two-Stage Exit (use_two_stage_exit): An advanced, toggleable exit strategy.

If False (the default), it uses a single profit target of 2.75R (2.75 times the initial risk).

If True, it sells half the position at a smaller, high-probability target of 1.25R, moves the stop-loss to breakeven, and lets the remaining half run with the trailing stop.

Risk & Position Sizing
risk_per_trade_percent: The strategy will only risk 2% of the total portfolio equity on any single trade.

use_dynamic_position_sizing: This is a crucial feature. It calculates the position size not just on the 2% rule, but also ensures that the total combined risk of all open positions does not exceed the max_portfolio_risk_percent (which is dynamically set by the market regime).

4. The Engine Room: run_backtest() Explained
This is the main function where the simulation happens. It's a chronological loop that ensures no future information is used to make decisions.

Load Data: It begins by loading all the necessary daily and 15-minute data files into memory for fast access.

The Daily Loop: It iterates through every single trading day in the backtest period.

Check Market Regime: On each day, the very first thing it does is check the market "weather" using the Market Regime Filter. This determines if it's even allowed to look for new trades.

Monitor Intraday Action: It then simulates the trading day by looping through the 15-minute intraday candles.

Manage Exits: It checks all open positions to see if any have hit their stop-loss or profit target.

Look for Entries: It checks the watchlist (which was created on the previous day) to see if any stocks have crossed their entry trigger price. If so, it executes a new trade.

End-of-Day Scan: After the market "closes," if the market regime is favorable, it scans all 200 stocks using the "Elite" entry filters to find potential setups for the next trading day.

Update Portfolio: At the very end of the day, it updates the total value of the portfolio and trails the stop-losses on any open positions.

Generate Report: After the loop has finished, it calculates all the final performance metrics (CAGR, Drawdown, Profit Factor, etc.) and saves the detailed logs to CSV files.

This strict, day-by-day process is the key to a realistic and bias-free backtest.

5. A Day in the Life: The Timeline of Execution
This section breaks down the precise, chronological order of events as the simulator processes a single trading day. Understanding this timeline is crucial to grasping how the system avoids lookahead bias and simulates a realistic trading environment.

Pre-Market (Before 9:15 AM IST)
For the Layman: Before the market opens, the simulator does its homework. It checks the overall market "weather" (the Market Regime) to decide if it's a good day to trade. It also reviews its "shopping list" (watchlist) of stocks that looked promising at the end of the previous day.

For the Trader: The simulation day begins. The script's first action is to evaluate the market_regime_filter using the closing data from the previous day (T-1). This determines the market_uptrend flag and sets the todays_max_portfolio_risk for the upcoming session. It then loads the watchlist for the current date, which contains all potential trade setups identified at the close of T-1.

For the Coder: The main for date in master_dates: loop advances to the current day. The code block for the market_regime_filter is executed, setting the market_uptrend boolean and regime_status string. The script then calls watchlist.get(date, {}) to retrieve the dictionary of potential setups for the current trading day.

Market Hours (9:15 AM - 3:30 PM IST)
For the Layman: As the trading day unfolds, the simulator watches the market like a hawk, but in fast-forward. It checks the price of stocks every 15 minutes. If a stock on its shopping list hits the target entry price, it "buys" it. At the same time, it's managing all the stocks it already owns, "selling" them if they hit their pre-defined profit or loss levels.

For the Trader: This is the event-driven core of the simulation. The script enters a nested loop that iterates through the 15-minute intraday candles for the current day.

Exit Management First: On every 15-minute candle, the first action is to check all open positions in portfolio['positions']. It compares the candle's high and low against each position's stop-loss and profit target to simulate exits. This is done first to accurately model a scenario where a stop is hit before a new trade is entered.

Entry Execution Second: After handling exits, the script checks the todays_watchlist. It compares the 15-minute candle's high against the trigger_price for each stock on the list. If triggered, the entry logic is called, a position is sized, and a new entry is added to the portfolio.

For the Coder: The for candle_time, _ in intraday_candles_today.iterrows(): loop begins. Inside this loop, the exit management logic is processed first, iterating through portfolio['positions']. Only after this loop completes does the code proceed to iterate through the todays_watchlist to check for new entry triggers. This strict sequential processing within the intraday loop is critical for simulation integrity.

Post-Market (After 3:30 PM IST)
For the Layman: Once the market closes, the simulator does its end-of-day accounting. It calculates the new total value of the portfolio. Then, it scans all 200 stocks again to create a new "shopping list" for the next day.

For the Trader: After the intraday loop is complete, two main EOD (End-of-Day) processes occur:

Portfolio Update & Trailing Stops: The script calculates the final closing equity for the day. It then iterates through all open positions and updates their stop_loss based on the use_atr_trailing_stop logic.

Scouting for T+1: The script then begins the "scouting" phase for the next day (T+1). It loops through all symbols and applies the full stack of entry filters (EMA, Wick, Volume, etc.) to the closing data of the current day (T). Any stock that passes all checks is added to the watchlist for the next calendar day.

For the Coder: After the intraday loop finishes, the portfolio equity is updated. Following this, the watchlist generation block runs, iterating through all symbols in daily_data. If a setup is identified, a new key-value pair is added to the main watchlist dictionary, where the key is the datetime object for the next trading day.

This detailed, time-based execution ensures that every decision is made using only the information that would have been available at that specific moment in time, providing a robust and realistic simulation.

6. From Backtest to Live Trading: The Next Steps
Transitioning a backtested strategy into a live, automated trading system is a significant undertaking that introduces new challenges. The backtester lives in a perfect world of historical data; a live system must deal with the complexities of the real market.

For the Layman 🤖
Think of the backtester as a flight simulator where a pilot practices. To actually fly a plane, the pilot needs a real plane (a broker), a connection to air traffic control (a data feed), and a flight plan (the trading logic). A live trading system is the real plane. It needs to connect to a stockbroker to place actual buy and sell orders, get live price information, and run automatically every day without your intervention. It also needs safety checks and alarms, just like a real cockpit.

For the Trader 🔩
The goal is to create a robust, fault-tolerant execution platform that can run the strategy autonomously. This requires several new components:

Broker Integration: The system needs to connect to a broker's API (Application Programming Interface) to programmatically send orders (e.g., limit, market, stop-loss), check order status, and get real-time account information like cash balance and open positions.

Real-Time Data Feed: The reliance on historical CSV files must be replaced with a live data feed. This can come from the broker's API or a dedicated data vendor. The system must be able to handle a continuous stream of price updates.

Order Management Logic: This is a complex module that handles the entire lifecycle of an order. It needs to place the initial entry order (often a Stop-Limit order to enter on a breakout), track if it gets filled, and if so, immediately place the corresponding Stop-Loss order. It also needs to handle order cancellations and modifications for the trailing stop.

State Management: The backtester's simple portfolio dictionary must be replaced with a robust state management system. This could be a database or a persistent file that tracks live positions, cash, and order statuses, ensuring the system can be restarted without losing track of its current state.

Scheduling and Automation: The entire process—from the EOD scan to intraday monitoring—needs to be automated. This is typically done using a scheduler (like cron on Linux or Windows Task Scheduler) that runs the different parts of the script at the correct times (e.g., run the EOD scan at 4:00 PM IST, run the intraday monitor from 9:15 AM to 3:30 PM IST).

Notifications and Monitoring: A live system cannot run silently. It needs a robust notification system (e.g., sending alerts via Telegram, email, or SMS) to inform you of critical events like trade executions, errors, or system failures.

For the Coder 🖥️
The monolithic backtesting script must be refactored into a modular, multi-component application.

Broker API Wrapper: Create a dedicated class or module that abstracts all interactions with the broker's API. This class would have methods like place_order(), get_open_positions(), get_live_price(), etc. This isolates the broker-specific code from your core strategy logic.

Data Handler: A new module responsible for fetching and processing real-time data. This could use WebSockets for a continuous stream of ticks or poll the API at regular intervals (e.g., every second).

Execution Engine: The core logic that takes a signal from your strategy, calculates the position size, and uses the Broker API Wrapper to place the appropriate orders. This module needs sophisticated error handling to manage API failures, rejected orders, and partial fills.

State Database: Implement a simple database (like SQLite) or a file-based solution (like JSON or Pickle) to persist the system's state. Before starting, the script should load its state (e.g., open positions) from this database, and it should save its state after every significant event.

Scheduler/Daemon: The application needs to be daemonized or run as a scheduled task. The EOD scan (scouting) would be one script scheduled to run after market close. The intraday execution (sniping) would be another script that runs continuously during market hours.

Logging and Alerting: Integrate a robust logging library (like Python's logging module) to log every action and error to a file. Integrate an alerting library or API service to send push notifications for critical events.