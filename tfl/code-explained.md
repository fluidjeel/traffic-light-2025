TFL Automated Trading System: Technical Implementation Guide
Document Version: 1.0
Applicable System Version: v3.0 and later (tfl_live_trader.py v1.4+)

1. Introduction
This document provides a detailed technical explanation of the software and cloud components that constitute the "TrafficLight-Manny" (TFL) automated trading system. Its purpose is to serve as a reference for developers and architects, offering a low-level "code explained" guide to the live trading application, the supporting AWS Lambda functions, and the DynamoDB data schema.

This guide assumes the reader is familiar with the high-level system architecture and the official strategy rules.

2. EC2 Live Trading Application (tfl_live_trader.py)
The core of the system is a single, stateful Python application that runs on an EC2 instance during market hours. It is designed as a monolithic but component-based application to ensure the lowest possible latency for all real-time operations.

2.1. Code Architecture & Components
The script is organized into several distinct classes, each with a specific responsibility, all orchestrated by the MainAppController.

MainAppController

Role: The central orchestrator and the brain of the application.

Key Responsibilities:

Initialization: At startup, it coordinates the entire setup sequence: fetching the daily bias from AWS, loading the correct strategy profile and data file, initializing the broker and AWS connectors, and setting up the logger.

Event Loop: It runs the main while True: loop that keeps the application alive during market hours.

Orchestration: It receives events (like a new tick or a completed 15-minute bar) and delegates the required actions to the appropriate specialized components. For example, after a bar closes, it calls the SignalGenerator (which is part of its own logic in the current implementation), gets the result, and passes it to the ExecutionHandler logic.

AWSConnector

Role: A dedicated client for all communication with AWS services.

Key Methods:

get_trading_bias(): Fetches the "LONGS_ONLY" / "SHORTS_ONLY" string from the SSM Parameter Store.

get_fyers_credentials(): Securely retrieves API keys from AWS Secrets Manager.

persist_trade(trade_object): Writes or updates a trade's state to the DynamoDB table. It handles the crucial conversion of Python floats to DynamoDB's Decimal type to prevent precision loss.

load_open_positions(): A placeholder for a production function that would query DynamoDB at startup to load the state of any open positions, making the system resilient to restarts.

FyersBrokerConnector

Role: A wrapper for the Fyers API and WebSocket, isolating all broker-specific logic.

Key Features:

Authentication: Manages the manual auth_code generation and the subsequent creation of a valid access_token.

WebSocket Management: It connects to the Fyers data socket in a separate thread (ws_thread), subscribes to the initial list of stocks, and passes every incoming tick to the MainAppController's on_tick callback function.

Dynamic Subscriptions: Includes methods (subscribe_to_instrument, unsubscribe_from_instrument) to dynamically manage the list of instruments being tracked, which is essential for options trading.

Order Execution: Contains the logic to place and modify orders. In PAPER_TRADING_MODE, it simulates these actions and logs them; in live mode, it would make the actual API calls.

OptionsInstrumentHandler

Role: Manages all the complexity related to options instrument selection.

Key Method: get_tradable_option_symbol(...)

Functionality: This is the logic that translates a signal on an underlying stock into a specific, tradable options contract. It encapsulates the rules for applying the 7-day expiry rollover and selecting the correct In-The-Money (ITM) strike based on the strategy's parameters. A production version of this would involve fetching the live options chain from the broker.

2.2. The Real-Time Event Flow (on_tick)
The heart of the live application is the on_tick method. This function is executed hundreds or thousands of times per second.

A new tick arrives from the WebSocket.

The on_tick method is called.

The last_known_prices dictionary is instantly updated.

The tick is immediately passed to the check_exits_on_tick method. This ensures that stop-loss and take-profit conditions for open positions are checked with the lowest possible latency. The logic for multi-stage trailing stops (ATR, Breakeven) is executed here before the SL/TP check.

3. AWS Lambda Functions
The system utilizes two serverless Lambda functions for all non-real-time, scheduled tasks.

3.1. pre_market_regime_lambda.py
Purpose: To act as the system's daily strategist, deciding whether the market environment is favorable for longs, shorts, or no trades.

Trigger: An AWS EventBridge schedule (cron job) set for 8:45 AM IST.

Core Logic:

Fetches the necessary daily OHLCV data for the NIFTY 50, INDIA VIX, and the full F&O stock universe from a data provider API (e.g., Fyers History API).

Calculates the required indicators: the 30-day and 100-day SMA for the NIFTY 50, and the 50-day SMA for all stocks.

Calculates the market breadth (the percentage of stocks above/below their 50-day SMA).

Applies the strategy's regime filter rules to determine the day's bias.

Writes the final string ("LONGS_ONLY", "SHORTS_ONLY", or "NO_TRADES") to the specified AWS SSM Parameter (/tfl/trading_bias).

Publishes a message to an SNS topic to notify the operator of the day's bias.

3.2. post_market_reporting_lambda.py
Purpose: To act as the system's end-of-day accountant, generating a daily performance report.

Trigger: An AWS EventBridge schedule (cron job) set for 4:00 PM IST.

Core Logic:

Connects to the TFL_Trades DynamoDB table.

Performs a query to find all trades that were closed on the current trading day. This requires a Global Secondary Index (GSI) on an attribute like exit_date.

Aggregates the pnl attribute of all closed trades to calculate the total Net P&L for the day.

Calculates other simple metrics like the number of trades, winners, and losers.

Formats this information into a human-readable summary.

Publishes the summary message to the SNS topic (TFL-Notifications) to be sent to the operator.

4. DynamoDB Table Schema: TFL_Trades
DynamoDB is used as a transactional, auditable log for every trade. Its NoSQL structure is ideal for storing the evolving state of a trade's lifecycle.

Table Name: TFL_Trades

Billing Mode: Pay Per Request (On-Demand)

4.1. Key Schema
Partition Key (HASH): trade_id (String)

Description: A unique identifier (UUID) generated by the application for each new trade. This key allows DynamoDB to efficiently retrieve all records related to a single trade.

Sort Key (RANGE): timestamp_utc (String)

Description: An ISO 8601 formatted timestamp of when the trade's state was recorded. Using a sort key allows us to store multiple items for the same trade_id, effectively creating a chronological log of its entire lifecycle (e.g., OPEN, SL_MODIFIED_1, SL_MODIFIED_2, CLOSED).

4.2. Example Item Structure (Trade Lifecycle)
Here is an example of the items that would be stored in DynamoDB for a single paper trade. Note how the status and timestamp_utc change, creating a complete audit trail.

Item 1: Trade Opening

{
  "trade_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "timestamp_utc": "2025-09-01T05:00:00.123Z",
  "status": "OPEN",
  "strategy_name": "TrafficLight-Manny-LONGS_LIVE",
  "trading_bias": "LONGS_ONLY",
  "underlying_symbol": "RELIANCE",
  "instrument_symbol": "NSE:RELIANCE25SEP2800CE",
  "direction": "LONG",
  "entry_price": "150.55",
  "quantity": 100,
  "initial_sl": "145.20",
  "current_sl": "145.20",
  "tp": "200.75",
  "initial_risk_value": "535.00",
  "entry_order_id": "PAPER-BO-...",
  "sl_order_id": "PAPER-SL-...",
  "is_paper_trade": true
}

Item 2: Stop-Loss is Modified by Trailing Stop Logic

{
  "trade_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "timestamp_utc": "2025-09-01T06:30:15.456Z",
  "status": "SL_MODIFIED",
  "current_sl": "155.80",
  "modification_reason": "ATR_TRAILING_STOP_UPDATE"
  // Note: Only the changed attributes need to be logged here, but for a full state log, we can log the entire object.
}
