# ------------------------------------------------------------------------------------------------
# live_trader.py - The Main Application Entry Point
# ------------------------------------------------------------------------------------------------
#
# This script is the conductor of the orchestra.
# - It parses command-line arguments (like --verbose-logging).
# - It initializes all the core components in the correct order.
# - It "wires" all the components together by subscribing them to the relevant
#   events on the Event Bus.
# - It runs a main loop that provides a real-time console dashboard and handles
#   graceful shutdown to persist the session state.
#
# ------------------------------------------------------------------------------------------------

import numpy as np
if not hasattr(np, 'NaN'): np.NaN = np.nan

import time
import argparse
import sys
import os
import pandas as pd

import config
from logger_service import LoggerService
from event_bus import EventBus
from data_handler import DataHandler
from strategy_engine import StrategyEngine
from portfolio_manager import PortfolioManager
from execution_simulator import ExecutionSimulator

class ConsoleDashboard:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.last_summary = {}
        self.event_bus.subscribe('PORTFOLIO_SUMMARY', self.update_summary)

    def update_summary(self, data):
        self.last_summary = data

    def display(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        summary = self.last_summary
        
        equity = summary.get('equity', 0)
        cash = summary.get('cash', 0)
        count = summary.get('open_positions_count', 0)
        pnl = summary.get('unrealized_pnl', 0)
        
        print("--- TFL Live Trading Dashboard ---")
        print(f"Equity: {equity:,.2f} | Cash: {cash:,.2f}")
        print(f"Open Positions: {count} | Unrealized PnL: {pnl:,.2f}")
        print("----------------------------------")
        print("(Press Ctrl+C to exit gracefully)")

def main():
    parser = argparse.ArgumentParser(description="TFL Live Paper Trading Bot")
    parser.add_argument('--verbose-logging', action='store_true', help="Enable detailed CSV logging for diagnostics.")
    args = parser.parse_args()

    logger = LoggerService(verbose_logging=args.verbose_logging)
    logger.log_console("INFO", "Initializing trading session...")
    event_bus = EventBus()

    try:
        if config.SYMBOL_SOURCE == 'CSV':
            logger.log_console("INFO", f"Loading symbols from CSV file: {config.CSV_FILE_PATH}")
            symbols_df = pd.read_csv(config.CSV_FILE_PATH)
            trading_symbols = [f"NSE:{symbol}-EQ" for symbol in symbols_df.iloc[:, 0].unique().tolist()]
            logger.log_console("SUCCESS", f"Loaded {len(trading_symbols)} unique symbols.")
        else:
            trading_symbols = config.TRADING_SYMBOLS
    except Exception as e:
        logger.log_console("FATAL", f"Failed to load symbols: {e}"); sys.exit(1)

    data_handler = DataHandler(trading_symbols, event_bus, logger)
    if not data_handler.connect():
        logger.log_console("FATAL", "Could not connect to data provider. Exiting."); sys.exit(1)

    portfolio_manager = PortfolioManager(event_bus, logger)
    # Inject data_handler into portfolio_manager for MTM calculations
    portfolio_manager.data_handler = data_handler 
    
    strategy_engine = StrategyEngine(trading_symbols, event_bus, logger, data_handler)
    execution_simulator = ExecutionSimulator(event_bus, logger, portfolio_manager)
    dashboard = ConsoleDashboard(event_bus)

    logger.log_console("INFO", "All components initialized. System is live.")
    
    try:
        while True:
            event_bus.publish('SYSTEM_STATUS_UPDATE')
            dashboard.display()
            time.sleep(5) 
    except KeyboardInterrupt:
        logger.log_console("INFO", "Shutdown signal received. Saving state...")
    finally:
        portfolio_manager.save_state()
        logger.shutdown()

if __name__ == "__main__":
    main()

