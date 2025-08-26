# backtester.py
#
# Description:
# This script simulates a trading strategy on a pre-processed master dataset.
# It processes signals from the 'all_signals_master.parquet' file, executes
# virtual trades, and calculates key performance metrics for a trading portfolio.
#
# INSTRUCTIONS:
# 1. Ensure you have the 'pyarrow' library installed: pip install pyarrow
# 2. Run this script after successfully generating and verifying the master file
#    using create_master_dataset.py and verify_master_dataset.py.
#
# This script is designed to be a simple, single-pass backtester for a
# pre-computed signal dataset, avoiding lookahead bias.

import pandas as pd
import os
import sys
import warnings

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# The project root directory is determined automatically to build all other file paths.
# FIXED: Hard-code the root directory to fix the file path issue.
ROOT_DIR = "D:\\algo-2025"

# The path to the master Parquet file
MASTER_FILE_PATH = os.path.join(ROOT_DIR, "data", "all_signals_master.parquet")

# ==============================================================================
# --- BACKTESTING ENGINE ---
# ==============================================================================

class Backtester:
    """
    Simulates a trading strategy based on pre-computed signals.
    """
    def __init__(self, start_date=None, end_date=None, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.portfolio = {}  # Tracks positions: {'symbol': {'shares': int, 'entry_price': float}}
        self.trades = []  # List to store completed trades
        self.current_datetime = None
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

    def execute_trade(self, row):
        """
        Executes a trade based on the signal in the current row.
        """
        signal = row['signal']
        symbol = row['symbol']
        entry_price = row['entry_price']
        sl = row['sl']
        
        # Check if there is already an open position for this symbol
        if symbol in self.portfolio:
            self.close_position(row, signal)
        
        # Open a new position if a valid signal exists
        if signal != 0:
            trade_type = 'LONG' if signal == 1 else 'SHORT'
            self.open_position(symbol, trade_type, entry_price, sl)
            print(f"[{row.name}] OPEN {trade_type} | Symbol: {symbol} | Price: {entry_price:.2f} | SL: {sl:.2f}")


    def open_position(self, symbol, trade_type, entry_price, sl):
        """
        Opens a new position in the portfolio.
        This is a simplified version assuming fixed lot sizes or capital allocation.
        """
        # For simplicity, we'll assume a fixed allocation of capital per trade.
        # This can be customized later.
        allocation = self.initial_capital * 0.05  # Allocate 5% of initial capital per trade
        
        if self.capital >= allocation:
            shares = allocation / entry_price
            self.portfolio[symbol] = {
                'shares': shares,
                'entry_price': entry_price,
                'sl': sl,
                'trade_type': trade_type,
                'open_datetime': self.current_datetime
            }
            self.capital -= allocation

    def close_position(self, row, signal=0):
        """
        Closes an existing position in the portfolio.
        """
        symbol = row['symbol']
        if symbol not in self.portfolio:
            return

        pos = self.portfolio[symbol]
        shares = pos['shares']
        entry_price = pos['entry_price']
        close_price = row['close']
        trade_type = pos['trade_type']
        
        pnl = 0
        if trade_type == 'LONG':
            pnl = (close_price - entry_price) * shares
        else: # SHORT
            pnl = (entry_price - close_price) * shares
            
        trade = {
            'symbol': symbol,
            'open_datetime': pos['open_datetime'],
            'close_datetime': self.current_datetime,
            'trade_type': trade_type,
            'entry_price': entry_price,
            'close_price': close_price,
            'pnl': pnl
        }
        self.trades.append(trade)
        self.capital += (pos['shares'] * entry_price) + pnl  # Update capital
        del self.portfolio[symbol]
        print(f"[{row.name}] CLOSE {trade_type} | Symbol: {symbol} | P&L: {pnl:.2f} | Capital: {self.capital:.2f}")


    def run_simulation(self):
        """
        Main function to run the backtesting simulation.
        """
        print("--- Starting Backtesting Simulation ---")
        print(f"Loading master file from: {MASTER_FILE_PATH}")
        
        if not os.path.exists(MASTER_FILE_PATH):
            print("❌ ERROR: Master file not found. Please run create_master_dataset.py first.")
            return

        master_df = pd.read_parquet(MASTER_FILE_PATH)
        
        # Filter by date range if provided
        if self.start_date:
            master_df = master_df[master_df.index >= self.start_date]
        if self.end_date:
            master_df = master_df[master_df.index <= self.end_date]
        
        if master_df.empty:
            print("❌ ERROR: No data found for the specified date range.")
            return

        # Ensure the DataFrame is sorted chronologically
        master_df.sort_index(inplace=True)
        
        # Main simulation loop
        for index, row in master_df.iterrows():
            self.current_datetime = index
            self.execute_trade(row)
        
        # Close any remaining open positions at the end of the simulation
        for symbol in list(self.portfolio.keys()):
            self.close_position(master_df.iloc[-1])
        
        self.analyze_results()
        print("\n--- Simulation Complete! ---")

    def analyze_results(self):
        """
        Calculates and prints key performance metrics.
        """
        if not self.trades:
            print("\nNo trades were executed during the simulation.")
            return
            
        trades_df = pd.DataFrame(self.trades)
        total_pnl = trades_df['pnl'].sum()
        num_trades = len(trades_df)
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = (len(winning_trades) / num_trades) * 100 if num_trades > 0 else 0
        
        net_profit = self.capital - self.initial_capital
        
        print("\n--- Backtest Results ---")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital:   ${self.capital:,.2f}")
        print(f"Net Profit:      ${net_profit:,.2f}")
        print(f"Total Trades:    {num_trades}")
        print(f"Winning Trades:  {len(winning_trades)}")
        print(f"Losing Trades:   {len(losing_trades)}")
        print(f"Win Rate:        {win_rate:.2f}%")
        print(f"Total P&L:       ${total_pnl:,.2f}")

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    backtester = Backtester()
    backtester.run_simulation()
