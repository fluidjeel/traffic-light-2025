# ------------------------------------------------------------------------------------------------
# portfolio_manager.py - The Accountant and State Manager
# ------------------------------------------------------------------------------------------------
#
# This component acts as the single source of truth for the paper trading account.
# - Manages cash, equity, and unrealized PnL.
# - Holds the list of all open positions.
# - Applies risk management rules before approving new trades.
# - UPGRADE: The "trade only once per day" rule is now controlled by a toggleable
#   flag (ALLOW_SAME_DAY_REENTRY) in the config.py file.
#
# ------------------------------------------------------------------------------------------------

import json
import os
from datetime import datetime
import pytz

import config

class PortfolioManager:
    def __init__(self, event_bus, logger, data_handler):
        self.event_bus = event_bus
        self.logger = logger
        self.data_handler = data_handler
        self.timezone = pytz.timezone(config.MARKET_TIMEZONE)

        self.initial_capital = config.INITIAL_CAPITAL
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.equity_cap = config.INITIAL_CAPITAL * config.EQUITY_CAP_FOR_RISK_CALC_MULTIPLE
        
        self.open_positions = {} # {symbol: {position_details}}
        self.closed_trades = []
        
        self.traded_today = set()
        self.current_trading_day = None
        
        self._load_state()
        self.event_bus.subscribe('SYSTEM_STATUS_UPDATE', self.on_status_update)

    def on_status_update(self, data):
        """
        Receives the current timestamp, checks for a new day,
        and then calculates and broadcasts the latest portfolio summary.
        """
        now = data['timestamp']
        self._check_for_new_day(now)
        self._update_equity()
        
        summary = {
            'equity': self.equity,
            'cash': self.cash,
            'open_positions_count': len(self.open_positions),
            'unrealized_pnl': self.calculate_unrealized_pnl()
        }
        self.event_bus.publish('PORTFOLIO_SUMMARY', summary)

    def _check_for_new_day(self, now):
        """If the date has changed, reset the list of traded symbols."""
        if self.current_trading_day != now.date():
            self.logger.log_console("INFO", f"New trading day: {now.date()}. Resetting daily trade tracker.")
            self.traded_today.clear()
            self.current_trading_day = now.date()

    def calculate_unrealized_pnl(self):
        """Calculates the PnL of all open positions based on the latest tick prices."""
        unrealized_pnl = 0
        for symbol, pos in self.open_positions.items():
            last_price = self.data_handler.last_tick_prices.get(symbol, pos['entry_price'])
            
            if pos['direction'] == 'LONG':
                pnl = (last_price - pos['entry_price']) * pos['quantity']
            else: # SHORT
                pnl = (pos['entry_price'] - last_price) * pos['quantity']
            unrealized_pnl += pnl
        return unrealized_pnl

    def _update_equity(self):
        """Updates equity based on cash and unrealized PnL."""
        self.equity = self.cash + self.calculate_unrealized_pnl()

    def check_trade_viability(self, symbol, direction, entry_price, sl_price):
        """
        Checks if a potential trade is viable based on all risk and strategy rules.
        Returns the calculated quantity if viable, otherwise returns 0.
        """
        # --- UPGRADE: Check for same-day re-entry only if the config flag is OFF ---
        if not config.ALLOW_SAME_DAY_REENTRY:
            if symbol in self.traded_today:
                return 0, "ALREADY_TRADED_TODAY"
            
        if symbol in self.open_positions:
            return 0, "ALREADY_OPEN"

        if len(self.open_positions) >= config.STRICT_MAX_OPEN_POSITIONS:
            return 0, "MAX_POSITIONS_REACHED"

        if direction == 'LONG':
            risk_per_share = entry_price - sl_price
        else: # SHORT
            risk_per_share = sl_price - entry_price
        
        if risk_per_share <= 0:
            return 0, "INVALID_RISK_PER_SHARE"

        equity_for_risk_calc = min(self.equity, self.equity_cap)
        total_current_risk_value = sum(pos.get('initial_risk_value', 0) for pos in self.open_positions.values())
        
        desired_risk_amount = equity_for_risk_calc * config.RISK_PER_TRADE_PCT
        available_risk_budget = (equity_for_risk_calc * config.MAX_TOTAL_RISK_PCT) - total_current_risk_value
        if available_risk_budget <= 0:
            return 0, "MAX_TOTAL_RISK_EXCEEDED"

        final_risk_amount = min(desired_risk_amount, available_risk_budget)
        quantity_by_risk = int(final_risk_amount / risk_per_share)
        
        capital_for_trade = equity_for_risk_calc * config.MAX_CAPITAL_PER_TRADE_PCT
        quantity_by_capital = int(capital_for_trade / entry_price)

        final_quantity = min(quantity_by_risk, quantity_by_capital)

        if final_quantity <= 0:
            return 0, "ZERO_QUANTITY_CALCULATED"

        return final_quantity, "APPROVED"

    def record_entry(self, symbol, direction, quantity, entry_price, sl_price, initial_risk_value):
        """Records a new open position."""
        if symbol in self.open_positions: return

        if direction == 'LONG':
            cost = quantity * entry_price
            self.cash -= cost

        self.open_positions[symbol] = {
            'direction': direction,
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_time': datetime.now(self.timezone),
            'sl': sl_price,
            'tp': 0,
            'initial_risk_value': initial_risk_value,
            'initial_risk_per_share': initial_risk_value / quantity if quantity > 0 else 0,
            'be_activated': False,
            'aggressive_ts_activated': False,
            'last_logged_sl': sl_price,
            'current_atr': 0
        }
        self.logger.log_to_csv('execution_log', {
            'timestamp': datetime.now(self.timezone), 'symbol': symbol, 'direction': direction,
            'status': 'ENTRY_RECORDED', 'details': f'Qty: {quantity} @ {entry_price:.2f}'
        })
        self.save_state()

    def record_exit(self, symbol, exit_price, reason):
        """Closes a position, records the trade, and adds the symbol to the traded_today set."""
        if symbol not in self.open_positions: return

        pos = self.open_positions.pop(symbol)
        
        self.traded_today.add(symbol)
        
        if pos['direction'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['quantity']
            self.cash += (pos['quantity'] * exit_price)
        else: # SHORT
            pnl = (pos['entry_price'] - exit_price) * pos['quantity']
            self.cash += (pos['quantity'] * pos['entry_price']) + pnl

        trade_log = {
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_time': pos['entry_time'].isoformat(),
            'exit_time': datetime.now(self.timezone).isoformat(),
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'quantity': pos['quantity'],
            'pnl': pnl,
            'exit_reason': reason
        }
        self.closed_trades.append(trade_log)
        self.logger.log_to_csv('trade_log', trade_log)
        self.save_state()

    def save_state(self):
        """Saves the current open positions and traded_today set to a JSON file."""
        try:
            state_to_save = {
                'open_positions': {},
                'traded_today': list(self.traded_today), # Convert set to list for JSON
                'current_trading_day': self.current_trading_day.isoformat() if self.current_trading_day else None
            }
            for symbol, pos in self.open_positions.items():
                pos_copy = pos.copy()
                pos_copy['entry_time'] = pos_copy['entry_time'].isoformat()
                state_to_save['open_positions'][symbol] = pos_copy

            with open(config.SESSION_STATE_FILE, 'w') as f:
                json.dump(state_to_save, f, indent=4)
        except Exception as e:
            self.logger.log_console("ERROR", f"Failed to save session state: {e}")

    def _load_state(self):
        """Loads session state from the JSON file if it exists."""
        if os.path.exists(config.SESSION_STATE_FILE):
            try:
                with open(config.SESSION_STATE_FILE, 'r') as f:
                    loaded_state = json.load(f)
                    
                    for symbol, pos in loaded_state.get('open_positions', {}).items():
                        pos['entry_time'] = datetime.fromisoformat(pos['entry_time'])
                        self.open_positions[symbol] = pos
                    
                    self.traded_today = set(loaded_state.get('traded_today', [])) # Convert list back to set
                    
                    day_str = loaded_state.get('current_trading_day')
                    self.current_trading_day = datetime.fromisoformat(day_str).date() if day_str else None

                if self.open_positions or self.traded_today:
                    self.logger.log_console("INFO", f"Loaded {len(self.open_positions)} open positions and {len(self.traded_today)} traded symbols from previous session.")
            except Exception as e:
                self.logger.log_console("ERROR", f"Failed to load session state: {e}")
                os.remove(config.SESSION_STATE_FILE)

