TFL Automated Trading System: Live Trading Management Guide
Document Version: 1.0
Applicable System Version: v3.0 and later
Companion Document: tfl-strategy-rules.md

1. Introduction
This document provides the official operational protocols for the management of the "TrafficLight-Manny" (TFL) automated trading system. While the tfl-strategy-rules.md document defines the what of the strategy, this guide defines the how—the day-to-day procedures, checks, and balances required to run the system in both live and paper trading modes.

The role of the human operator is not to interfere with the system's core logic, but to act as a diligent supervisor, ensuring the automated infrastructure is functioning as designed and intervening only in predefined emergency scenarios. The primary directive is to let the system trade according to its rules.

2. Live Trading Protocol
When the system is executing trades with real capital (PAPER_TRADING_MODE = False), operational discipline is paramount. The following protocol outlines the daily responsibilities of the system operator.

2.1. Pre-Market Checklist (8:45 AM - 9:10 AM IST)
The operator must verify the system's readiness before the market opens.

Confirm System Readiness:

Verify that the AWS EventBridge rule has successfully started the EC2 trading server. The instance state in the EC2 console should be "Running".

Confirm receipt of the SNS notification (via email or SMS) from the pre-market Lambda. This notification contains the day's official trading bias (e.g., "LONGS_ONLY", "SHORTS_ONLY", or "NO_TRADES"). This is your most important piece of information for the day.

Verify Application Startup:

SSH into the EC2 instance.

Check the application log to confirm that the tfl_live_trader.py script has started and logged its initial messages successfully. The log should explicitly state which strategy profile it has loaded.

Command: tail -f /var/log/tfl_trader.log

Broker Terminal Check:

Log in to your broker's trading terminal (e.g., Fyers Web).

Ensure there are no open positions from previous days that the system might not be aware of (this should not happen in a correctly functioning system but is a crucial safety check).

Confirm your account balance and available margin.

2.2. Market Hours Monitoring (9:15 AM - 3:30 PM IST)
The operator's role during market hours is passive observation, not active intervention.

Primary Task: Monitor Logs: Keep a view of the application logs in AWS CloudWatch. The system is designed to be verbose. You should see informational messages about signals being processed, orders being placed, and stops being modified. Your primary responsibility is to watch for any logs with the level ERROR or FATAL.

Secondary Task: Monitor Positions: Periodically check the "Positions" tab in your broker's terminal. The positions displayed there should perfectly match the open positions being reported in the application's logs. Any discrepancy could indicate a problem with the broker's API response handling.

DO NOT Interfere: It is critical to not manually close a position that the system has opened, even if it is in a loss. The system's statistical edge is built on its ability to manage trades according to its predefined exit logic. Manual intervention invalidates the strategy.

2.3. Emergency Intervention Protocol
Manual intervention is only permitted in specific, pre-defined emergency scenarios that threaten the integrity of the entire system.

System Crash / Loss of Connectivity:

Detection: No new logs are appearing in CloudWatch for an extended period (e.g., > 5 minutes), or a CloudWatch Alarm for CPU/Network inactivity is triggered.

Action Protocol:

Attempt to SSH into the EC2 instance to diagnose the issue.

If the instance is unresponsive, perform a Reboot from the AWS EC2 console.

IMMEDIATELY log in to your broker's terminal. Manually place a market order to close ALL open positions that the bot had initiated. This is a critical safety measure to flatten your exposure while the system is offline.

Once the system is back online, it will load its state from DynamoDB, see that it has no open positions, and resume normal operations.

Broker-Wide API Outage:

Detection: The application logs will be filled with ERROR messages related to API connection failures. You will likely also see an announcement from your broker.

Action Protocol:

Log in to your broker's mobile or web terminal.

Manually square off all open positions.

SSH into the EC2 instance and stop the tfl_live_trader.py application to prevent it from sending more failing API calls. Do not trade for the rest of the day.

3. Paper Trading Protocol
Paper trading is the final validation stage. Its purpose is to test the entire software stack and strategy logic against live market data without financial risk.

Setup: Ensure the PAPER_TRADING_MODE flag in the tfl_live_trader.py script is set to True.

Daily Process: The daily operational protocol for paper trading is identical to live trading. The operator must perform the same pre-market checks, monitoring, and post-market reviews. This builds the necessary operational discipline.

Validation Checklist (End of Day): The primary goal is analysis. At the end of each paper trading day, the operator must review the DynamoDB logs and CloudWatch logs to answer the following questions:

Signal Integrity: Did the system generate signals on the correct 15-minute bars as per the strategy rules?

Execution Integrity: Was a paper trade correctly initiated and logged to DynamoDB for each valid, prioritized signal?

Options Selection: Was the correct ITM option contract symbol selected and logged?

Trade Management: Were trailing stop-loss modifications logged at the correct price points based on the live ticks?

Exit Integrity: Were paper exits triggered and logged correctly when the live price of the instrument touched the virtual SL/TP levels?

P&L Calculation: Does the P&L logged for each paper trade accurately reflect the difference between the logged entry and exit prices?

4. Options-Specific Instructions
When TRADE_INSTRUMENT_TYPE is set to 'OPTION', the following principles apply:

Signal Source: All entry signals are generated from the price action of the underlying stock (spot price). The system never uses the option's chart to find signals.

Strike Selection: The system is hard-coded to select the monthly contract that is 2 strikes In-The-Money (ITM). For long (call) positions, this means two strikes below the spot price. For short (put) positions, this means two strikes above the spot price. The target Delta is between 0.60 and 0.75.

Expiry Management: The system must automatically apply the 7-day rollover rule. If the current month's contract expires within the next 7 calendar days, the system will automatically select a strike from the next month's contract series.

Monitoring: The operator should see dynamic subscription messages in the logs. When a new options trade is initiated, a log entry should confirm the system has subscribed to the WebSocket for that specific option contract. When the trade is closed, a corresponding unsubscribe message should appear.

5. Potential Areas for Future Enhancement
The v3.0 system is robust, but there are always opportunities for improvement. The following are logical next steps for future development versions:

Smart Risk Resizing: Enhance the ExecutionHandler to not just reject a trade that exceeds the total risk budget, but to attempt to resize it to precisely fit the remaining available risk.

Partial Profit Taking: Introduce a mechanism to exit a portion of a position at a closer target (e.g., 2R or 3R) to smooth the equity curve, while letting the remainder of the position run towards the full 10R target.

Dynamic Regime Filters: Enhance the pre-market Lambda to use more dynamic thresholds for market breadth or volatility (e.g., based on a moving average of the VIX) instead of the current fixed values.

Multi-Strategy Portfolio: The ultimate evolution of the engine would be to allow it to load multiple, uncorrelated strategy profiles (e.g., a mean-reversion strategy alongside the current momentum one) and manage a single, blended portfolio.

Advanced Order Types: As broker APIs evolve, the system could be enhanced to use more sophisticated execution algorithms like TWAP (Time-Weighted Average Price) or VWAP (Volume-Weighted Average Price) for entering large positions to minimize market impact.