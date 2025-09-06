# ------------------------------------------------------------------------------------------------
# execution_simulator.py - A Professional-Grade Order Management Simulator
# ------------------------------------------------------------------------------------------------
#
# ARCHITECTURAL UPGRADE: This component now operates as a sophisticated Order Manager,
# perfectly replicating the proactive, time-bound order logic you proposed.
#
# CRITICAL NEW FEATURE: The exit management logic is now complete and mirrors the
# backtest, including the multi-stage aggressive ATR trailing stop.
#
# CRITICAL NEW FEATURE: Every single modification to a stop-loss is now logged to
# a dedicated 'in_trade_management_log.csv' for complete transparency.
#
# ------------------------------------------------------------------------------------------------

import time
from datetime import datetime
import random

import config

class ExecutionSimulator:
    def __init__(self, event_bus, logger, portfolio_manager):
        self.event_bus = event_bus
        self.logger = logger
        self.portfolio_manager = portfolio_manager
        
        self.pending_orders = {} # {symbol: {order_details}}
        
        self.event_bus.subscribe('PotentialTradeSignal', self.on_new_signal)
        self.event_bus.subscribe('TICK', self.on_tick)
        self.event_bus.subscribe('CANDLE_CLOSED_15MIN_WITH_ATR', self.on_15min_candle_with_atr)

    def on_new_signal(self, data):
        """
        Receives a valid setup from the Strategy Engine and simulates
        placing a stop order with the broker.
        """
        symbol = data['symbol']
        if symbol in self.portfolio_manager.open_positions or symbol in self.pending_orders:
            return

        self.pending_orders[symbol] = {
            'direction': data['direction'],
            'trigger_price': data['trigger_price'],
            'sl_price': data['sl_price'],
            'creation_time': datetime.now(self.portfolio_manager.timezone)
        }
        self.logger.log_to_csv('execution_log', {
            'timestamp': datetime.now(self.portfolio_manager.timezone),
            'symbol': symbol, 'direction': data['direction'],
            'status': 'STOP_ORDER_PENDING', 'details': f"Awaiting price to cross {data['trigger_price']:.2f}"
        })

    def on_tick(self, tick):
        """
        Acts like the broker's server. Checks every tick against our book of
        pending stop orders and open positions.
        """
        symbol = tick['symbol']
        price = tick['price']

        if symbol in self.pending_orders:
            order = self.pending_orders[symbol]
            triggered = False
            if order['direction'] == 'LONG' and price >= order['trigger_price']:
                triggered = True
            elif order['direction'] == 'SHORT' and price <= order['trigger_price']:
                triggered = True
            
            if triggered:
                self._execute_entry(symbol, order, market_price=price)
                del self.pending_orders[symbol]

        if symbol in self.portfolio_manager.open_positions:
            self._manage_open_position(symbol, price)

    def _execute_entry(self, symbol, order, market_price):
        """
        Simulates the execution of a triggered stop order using the
        actual market price that caused the trigger.
        """
        direction = order['direction']
        sl_price = order['sl_price']

        base_price = market_price
        
        slippage = base_price * config.SLIPPAGE_FACTOR * (random.random() - 0.5) * 2
        fill_price = base_price + slippage

        time.sleep(config.SIMULATED_LATENCY_MS / 1000.0)

        quantity, reason = self.portfolio_manager.check_trade_viability(symbol, direction, fill_price, sl_price)

        if quantity > 0:
            initial_risk_value = (abs(fill_price - sl_price)) * quantity
            self.portfolio_manager.record_entry(symbol, direction, quantity, fill_price, sl_price, initial_risk_value)
            
            self.logger.log_console("FILL", f"[{direction}] {symbol} - Filled {quantity} units @ {fill_price:.2f}")
            self.logger.log_to_csv('execution_log', {
                'timestamp': datetime.now(self.portfolio_manager.timezone), 'symbol': symbol, 'direction': direction,
                'status': 'ENTRY_FILLED', 'details': f"Qty: {quantity} @ {fill_price:.2f}"
            })
        else:
            self.logger.log_console("REJECT", f"[{direction}] {symbol} - Trade rejected. Reason: {reason}")
            self.logger.log_to_csv('execution_log', {
                'timestamp': datetime.now(self.portfolio_manager.timezone), 'symbol': symbol, 'direction': direction,
                'status': 'REJECTED', 'details': f"Reason: {reason}"
            })

    def on_15min_candle_with_atr(self, data):
        """
        This method now serves two purposes:
        1. Cancels any stale pending orders from the previous candle.
        2. Updates the ATR for open positions for the trailing stop.
        """
        candle_close_time = data['candle']['datetime']

        for symbol, order in list(self.pending_orders.items()):
            if order['creation_time'] < candle_close_time:
                del self.pending_orders[symbol]
                self.logger.log_console("CANCEL", f"Order for {symbol} cancelled (not filled in time).")
                self.logger.log_to_csv('execution_log', {
                    'timestamp': datetime.now(self.portfolio_manager.timezone),
                    'symbol': symbol, 'direction': order['direction'],
                    'status': 'ORDER_CANCELLED', 'details': 'Not filled within 15min entry window'
                })

        symbol = data['symbol']
        if symbol in self.portfolio_manager.open_positions:
            self.portfolio_manager.open_positions[symbol]['current_atr'] = data['atr']

    def _manage_open_position(self, symbol, price):
        """
        Manages the complete, multi-stage exit strategy for an open position
        and logs every SL modification for verification.
        """
        pos = self.portfolio_manager.open_positions[symbol]
        original_sl = pos['sl']
        
        # --- Calculate Current R-value (Profit in terms of Initial Risk) ---
        initial_risk_per_share = pos.get('initial_risk_per_share', 0)
        current_r = 0
        if initial_risk_per_share > 0:
            if pos['direction'] == 'LONG':
                current_r = (price - pos['entry_price']) / initial_risk_per_share
            else: # SHORT
                current_r = (pos['entry_price'] - price) / initial_risk_per_share

        # --- 1. Breakeven Logic ---
        if not pos.get('be_activated', False) and current_r >= config.BREAKEVEN_TRIGGER_R:
            new_sl = 0
            if pos['direction'] == 'LONG':
                new_sl = pos['entry_price'] + (initial_risk_per_share * config.BREAKEVEN_PROFIT_R)
            else: # SHORT
                new_sl = pos['entry_price'] - (initial_risk_per_share * config.BREAKEVEN_PROFIT_R)
            
            if new_sl != pos['sl']:
                pos['sl'] = new_sl
                pos['be_activated'] = True
                self.logger.log_to_csv('in_trade_management_log', {
                    'timestamp': datetime.now(self.portfolio_manager.timezone), 'symbol': symbol, 
                    'update_type': 'BREAKEVEN_TRIGGERED', 'old_sl': original_sl, 'new_sl': new_sl, 
                    'current_price': price, 'current_r_value': f"{current_r:.2f}R"
                })

        # --- 2. Multi-Stage ATR Trailing Stop ---
        atr = pos.get('current_atr', 0)
        if atr > 0:
            atr_multiplier = 0
            update_type = 'ATR_TRAIL_UPDATE'
            
            if pos['direction'] == 'LONG':
                # Check if aggressive trailing should be activated
                if current_r >= config.LONG_AGGRESSIVE_TS_TRIGGER_R:
                    atr_multiplier = config.LONG_AGGRESSIVE_TS_MULTIPLIER
                    if not pos.get('aggressive_ts_activated', False):
                        update_type = 'AGGRESSIVE_TRAIL_ACTIVATED'
                        pos['aggressive_ts_activated'] = True
                else:
                    atr_multiplier = config.LONG_ATR_TS_MULTIPLIER
                
                new_tsl = price - (atr * atr_multiplier)
                pos['sl'] = max(pos['sl'], new_tsl) # Ensure SL only moves up

            else: # SHORT
                if current_r >= config.SHORT_AGGRESSIVE_TS_TRIGGER_R:
                    atr_multiplier = config.SHORT_AGGRESSIVE_TS_MULTIPLIER
                    if not pos.get('aggressive_ts_activated', False):
                        update_type = 'AGGRESSIVE_TRAIL_ACTIVATED'
                        pos['aggressive_ts_activated'] = True
                else:
                    atr_multiplier = config.SHORT_ATR_TS_MULTIPLIER
                
                new_tsl = price + (atr * atr_multiplier)
                pos['sl'] = min(pos['sl'], new_tsl) # Ensure SL only moves down

            # Log if the trailing stop moved the SL
            if pos['sl'] != original_sl and pos['sl'] != pos.get('last_logged_sl', 0):
                self.logger.log_to_csv('in_trade_management_log', {
                    'timestamp': datetime.now(self.portfolio_manager.timezone), 'symbol': symbol, 
                    'update_type': update_type, 'old_sl': original_sl, 'new_sl': pos['sl'], 
                    'current_price': price, 'current_r_value': f"{current_r:.2f}R"
                })
                pos['last_logged_sl'] = pos['sl']

        # --- 3. Check for SL/TP Hit ---
        exit_price, exit_reason = None, None
        if pos['direction'] == 'LONG':
            if price <= pos['sl']: exit_price, exit_reason = pos['sl'], 'SL_HIT'
            elif pos.get('tp', 0) > 0 and price >= pos['tp']: exit_price, exit_reason = pos['tp'], 'TP_HIT'
        else: # SHORT
            if price >= pos['sl']: exit_price, exit_reason = pos['sl'], 'SL_HIT'
            elif pos.get('tp', 0) > 0 and price <= pos['tp']: exit_price, exit_reason = pos['tp'], 'TP_HIT'

        if exit_reason:
            self.portfolio_manager.record_exit(symbol, exit_price, exit_reason)
            self.logger.log_console("EXIT", f"[{pos['direction']}] {symbol} - Closed @ {exit_price:.2f}. Reason: {exit_reason}")

