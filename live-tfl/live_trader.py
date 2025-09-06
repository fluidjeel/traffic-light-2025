# ------------------------------------------------------------------------------------------------
# live_trader.py - The Main Entry Point and Conductor
# ------------------------------------------------------------------------------------------------
#
# This script is the heart of the application. It performs the following steps:
# 1. Sets up the environment (like the NumPy fix for pandas_ta).
# 2. Parses command-line arguments (e.g., --verbose-logging).
# 3. Initializes all the core components.
# 4. Initializes a SystemHealthMonitor to detect silent data feed failures.
# 5. Connects the DataHandler and triggers a "Startup Sanity Check".
# 6. Runs the main application loop for real-time monitoring and health checks.
#
# ------------------------------------------------------------------------------------------------

import time
import argparse
import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz

np.NaN = np.nan

from logger_service import LoggerService
from event_bus import EventBus
from data_handler import DataHandler
from strategy_engine import StrategyEngine
from portfolio_manager import PortfolioManager
from execution_simulator import ExecutionSimulator
import config

logger = None
data_handler = None
keep_running = True

class Dashboard:
    def __init__(self, event_bus, logger):
        self.event_bus = event_bus
        self.logger = logger
        self.last_summary = {}
        self.event_bus.subscribe('PORTFOLIO_SUMMARY', self.on_summary)

    def on_summary(self, summary):
        self.last_summary = summary

    def display(self, now):
        equity = self.last_summary.get('equity', config.INITIAL_CAPITAL)
        cash = self.last_summary.get('cash', config.INITIAL_CAPITAL)
        pnl = self.last_summary.get('unrealized_pnl', 0)
        pos_count = self.last_summary.get('open_positions_count', 0)
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n--- TFL Live Trading Dashboard ---")
        print(f"Equity: {equity:,.2f} | Cash: {cash:,.2f} | Unrealized PnL: {pnl:,.2f}")
        print(f"Open Positions: {pos_count} | Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print("----------------------------------")
        print("(Press Ctrl+C to exit gracefully)")

class SystemHealthMonitor:
    """A watchdog to monitor the health of the WebSocket data feed."""
    def __init__(self, event_bus, logger, timezone):
        self.event_bus = event_bus
        self.logger = logger
        self.timezone = timezone
        self.last_heartbeat_time = datetime.now(self.timezone)
        self.event_bus.subscribe('DATA_HANDLER_HEARTBEAT', self.on_heartbeat)

    def on_heartbeat(self, data):
        """Updates the timestamp of the last received heartbeat."""
        self.last_heartbeat_time = data['timestamp']

    def check(self):
        """
        Checks if the data feed is still alive. If not, triggers a shutdown.
        Returns False if the system should shut down, True otherwise.
        """
        now = datetime.now(self.timezone)
        time_since_last_heartbeat = now - self.last_heartbeat_time
        if time_since_last_heartbeat.total_seconds() > config.WEBSOCKET_HEARTBEAT_THRESHOLD:
            self.logger.log_console("FATAL", "Data feed appears to be down (no heartbeat received). Shutting down.")
            return False
        return True

def load_symbols():
    if config.SYMBOL_SOURCE == 'CSV':
        try:
            df = pd.read_csv(config.CSV_FILE_PATH)
            symbols = ["NSE:" + s + "-EQ" for s in df['symbol'].unique()]
            logger.log_console("SUCCESS", f"Loaded {len(symbols)} unique symbols from {config.CSV_FILE_PATH}.")
            return symbols
        except Exception as e:
            logger.log_console("FATAL", f"Failed to load symbols from CSV: {e}")
            sys.exit(1)
    else:
        logger.log_console("INFO", "Loading symbols from manual list in config.")
        return config.TRADING_SYMBOLS

def main():
    global logger, data_handler, keep_running
    
    parser = argparse.ArgumentParser(description="TFL Live Paper Trading Bot")
    parser.add_argument('--verbose-logging', action='store_true', help="Enable detailed CSV logging.")
    args = parser.parse_args()

    logger = LoggerService(verbose=args.verbose_logging)
    logger.log_console("INFO", "Initializing trading session...")
    
    event_bus = EventBus()
    symbols = load_symbols()
    timezone = pytz.timezone(config.MARKET_TIMEZONE)

    data_handler = DataHandler(symbols, event_bus, logger)
    if not data_handler.connect_and_load_history():
        logger.log_console("FATAL", "Could not connect or load historical data. Exiting.")
        sys.exit(1)

    portfolio_manager = PortfolioManager(event_bus, logger, data_handler)
    strategy_engine = StrategyEngine(symbols, event_bus, logger, data_handler)
    execution_simulator = ExecutionSimulator(event_bus, logger, portfolio_manager)
    dashboard = Dashboard(event_bus, logger)
    health_monitor = SystemHealthMonitor(event_bus, logger, timezone)

    logger.log_console("INFO", "All components initialized and wired.")

    strategy_engine.run_sanity_check()

    try:
        while keep_running:
            now = datetime.now(timezone)
            # --- UPGRADE: Pass timestamp to the status update event ---
            event_bus.publish('SYSTEM_STATUS_UPDATE', {'timestamp': now})
            dashboard.display(now)
            
            if not health_monitor.check():
                keep_running = False
            
            time.sleep(5)
    except KeyboardInterrupt:
        logger.log_console("INFO", "Ctrl+C detected. Shutting down gracefully...")
        keep_running = False

    if data_handler: data_handler.stop()
    if portfolio_manager: portfolio_manager.save_state()
    if logger: logger.shutdown()
    print("\nSystem shut down complete.")

if __name__ == "__main__":
    main()

