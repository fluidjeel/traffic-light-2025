# ------------------------------------------------------------------------------------------------
# portfolio_manager.py - The Central State Manager for the Paper Trading Account
# ------------------------------------------------------------------------------------------------
#
# This component acts as the "accountant" and single source of truth for the portfolio.
# - It tracks cash, equity, open positions, and total risk exposure.
# - It provides methods for other components to open, close, and query positions.
# - It enforces all portfolio-level risk rules (max positions, max total risk, etc.).
# - It handles state persistence, saving open positions to a file on shutdown and
#   reloading them on startup.
# - It broadcasts a summary of its state for real-time monitoring.
#
# ------------------------------------------------------------------------------------------------

import json
import os
from datetime import datetime

import config

class PortfolioManager:
    def __init__(self, event_bus, logger):
        self.event_bus = event_bus
        self.logger = logger
        
        self.cash = config.INITIAL_CAPITAL
        self.equity = config.INITIAL_CAPITAL
        self.open_positions = {} # {symbol: {position_details}}
        self.equity_cap = config.INITIAL_CAPITAL * config.EQUITY_CAP_FOR_RISK_CALC_MULTIPLE
        
        self._load_state() # Load previous session state on startup
        
        self.event_bus.subscribe('TICK', self.on_tick_for_pnl)
        self.event_bus.subscribe('SYSTEM_STATUS_UPDATE', self.broadcast_summary)

    def _load_state(self):
        """Loads open positions from the session state file if it exists."""
        if os.path.exists(config.SESSION_STATE_FILE):
            try:
                with open(config.SESSION_STATE_FILE, 'r') as f:
                    saved_positions = json.load(f)
                    self.open_positions = saved_positions
                    self.logger.log_console("SUCCESS", f"Loaded {len(self.open_positions)} open positions from previous session.")
            except Exception as e:
                self.logger.log_console("ERROR", f"Could not load session state: {e}")

    def save_state(self):
        """Saves the current open positions to the session state file."""
        try:
            with open(config.SESSION_STATE_FILE, 'w') as f:
                json.dump(self.open_positions, f, indent=4)
            self.logger.log_console("INFO", f"Saved {len(self.open_positions)} open positions to session file.")
        except Exception as e:
            self.logger.log_console("ERROR", f"Could not save session state: {e}")

    def on_tick_for_pnl(self, data):
        """Updates the real-time equity of the portfolio on every tick."""
        # This is a simplified MTM calculation. A real system would be more complex.
        market_value_of_positions = 0
        for symbol, pos in self.open_positions.items():
            current_price = self.data_handler.last_tick_prices.get(symbol, pos['entry_price'])
            if pos['direction'] == 'LONG':
                market_value_of_positions += pos['quantity'] * current_price
            else: # SHORT
                # PnL for shorts is more complex to model simply.
                # This is an approximation of the position's value change.
                pnl = (pos['entry_price'] - current_price) * pos['quantity']
                market_value_of_positions += (pos['entry_price'] * pos['quantity']) + pnl
        
        self.equity = self.cash + market_value_of_positions

    def open_position(self, symbol, direction, entry_price, sl_price):
        """The main method for opening a new position after checking all rules."""
        # --- 1. Check Portfolio Constraints ---
        if len(self.open_positions) >= config.STRICT_MAX_OPEN_POSITIONS:
            self.logger.log_to_csv('execution_log', {'symbol': symbol, 'status': 'REJECTED', 'rejection_reason': 'MAX_POSITIONS_REACHED'})
            return

        # --- 2. Calculate Position Size ---
        equity_for_risk_calc = min(self.equity, self.equity_cap)
        
        # Determine available risk budget
        total_current_risk_value = sum(pos.get('initial_risk_value', 0) for pos in self.open_positions.values())
        available_risk_budget = (equity_for_risk_calc * config.MAX_TOTAL_RISK_PCT) - total_current_risk_value
        
        if available_risk_budget <= 0:
            self.logger.log_to_csv('execution_log', {'symbol': symbol, 'status': 'REJECTED', 'rejection_reason': 'MAX_TOTAL_RISK_EXCEEDED'})
            return

        # Calculate risk for this trade
        desired_risk_amount = equity_for_risk_calc * config.RISK_PER_TRADE_PCT
        risk_amount_for_trade = min(desired_risk_amount, available_risk_budget)
        
        risk_per_share = abs(entry_price - sl_price)
        if risk_per_share == 0: return

        quantity_by_risk = int(risk_amount_for_trade / risk_per_share)
        
        # Check capital concentration limit
        capital_for_trade = equity_for_risk_calc * config.MAX_CAPITAL_PER_TRADE_PCT
        quantity_by_capital = int(capital_for_trade / entry_price)
        
        final_quantity = min(quantity_by_risk, quantity_by_capital)

        if final_quantity <= 0:
            self.logger.log_to_csv('execution_log', {'symbol': symbol, 'status': 'REJECTED', 'rejection_reason': 'ZERO_QUANTITY_CALCULATED'})
            return

        # --- 3. Execute and Record Position ---
        cost_of_trade = final_quantity * entry_price
        self.cash -= cost_of_trade # Simplified cash management

        initial_risk_value = final_quantity * risk_per_share
        
        if direction == 'LONG':
            tp_price = entry_price + (risk_per_share * config.RISK_REWARD_RATIO)
            multiplier = config.ATR_TS_MULTIPLIER_LONG
        else: # SHORT
            tp_price = entry_price - (risk_per_share * config.RISK_REWARD_RATIO)
            multiplier = config.ATR_TS_MULTIPLIER_SHORT

        self.open_positions[symbol] = {
            'direction': direction, 'entry_price': entry_price, 'quantity': final_quantity,
            'sl': sl_price, 'tp': tp_price, 'entry_time': datetime.now().isoformat(),
            'initial_risk_per_share': risk_per_share,
            'initial_risk_value': initial_risk_value,
            'be_activated': False,
            'current_ts_multiplier': multiplier
        }
        
        self.logger.log_console("FILL", f"[{direction}] {symbol} - Filled {final_quantity} units @ {entry_price:.2f}. SL={sl_price:.2f}, TP={tp_price:.2f}")
        self.event_bus.publish('FILL', self.open_positions[symbol])
        self.logger.log_to_csv('execution_log', {'symbol': symbol, 'direction': direction, 'status': 'FILLED', 'final_qty': final_quantity, 'fill_price': entry_price})

    def close_position(self, symbol, exit_price, reason):
        """Closes an open position and updates portfolio metrics."""
        if symbol not in self.open_positions: return

        pos = self.open_positions.pop(symbol)
        pnl = (exit_price - pos['entry_price']) * pos['quantity'] if pos['direction'] == 'LONG' else (pos['entry_price'] - exit_price) * pos['quantity']
        
        self.cash += (pos['quantity'] * exit_price)
        self.equity += pnl

        self.logger.log_console("EXIT", f"[{pos['direction']}] {symbol} - Closed @ {exit_price:.2f}. Reason: {reason}. PnL: {pnl:,.2f}")
        self.event_bus.publish('EXIT', {'position': pos, 'pnl': pnl})

    def get_position(self, symbol):
        return self.open_positions.get(symbol)
        
    def broadcast_summary(self, _=None):
        """Publishes a summary of the current portfolio state."""
        total_pnl = sum(
            (self.data_handler.last_tick_prices.get(s, p['entry_price']) - p['entry_price']) * p['quantity'] if p['direction'] == 'LONG' 
            else (p['entry_price'] - self.data_handler.last_tick_prices.get(s, p['entry_price'])) * p['quantity']
            for s, p in self.open_positions.items()
        )
        
        summary = {
            'equity': self.equity, 'cash': self.cash,
            'open_positions_count': len(self.open_positions),
            'unrealized_pnl': total_pnl
        }
        self.event_bus.publish('PORTFOLIO_SUMMARY', summary)

