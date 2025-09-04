# ------------------------------------------------------------------------------------------------
# execution_simulator.py - Simulates Order Execution and Manages Trade Exits
# ------------------------------------------------------------------------------------------------
#
# This component acts as the "trader". It's responsible for all order-related logic.
# - Listens for 'PotentialTradeSignal' events from the Strategy Engine.
# - Queries the Portfolio Manager to check risk and capital limits before entering a trade.
# - Simulates trade entries with realistic latency and slippage.
# - Continuously monitors live ticks to check for exits (SL, TP, Trailing Stop).
# - Implements the full dynamic exit strategy:
#   1. Breakeven Stop
#   2. Multi-stage ATR Trailing Stop
# - Publishes 'FILL' and 'EXIT' events.
#
# ------------------------------------------------------------------------------------------------

import time
import random

import config

class ExecutionSimulator:
    def __init__(self, event_bus, logger, portfolio_manager):
        self.event_bus = event_bus
        self.logger = logger
        self.portfolio_manager = portfolio_manager
        
        self.pending_triggers = {} # {symbol: {'direction': 'LONG', 'trigger_price': X, 'sl_price': Y}}
        
        self.event_bus.subscribe('PotentialTradeSignal', self.on_potential_signal)
        self.event_bus.subscribe('TICK', self.on_tick)
        self.event_bus.subscribe('CANDLE_CLOSED_15MIN', self.on_15min_candle_for_tsl)

    def on_potential_signal(self, data):
        """Stores a signal from the strategy engine, waiting for a tick to trigger it."""
        self.pending_triggers[data['symbol']] = data

    def on_tick(self, data):
        """Main tick handler for checking entry triggers and exit conditions."""
        symbol = data['symbol']
        price = data['price']

        # --- 1. Check for ENTRY Triggers ---
        if symbol in self.pending_triggers:
            trigger_data = self.pending_triggers[symbol]
            
            if (trigger_data['direction'] == 'LONG' and price >= trigger_data['trigger_price']) or \
               (trigger_data['direction'] == 'SHORT' and price <= trigger_data['trigger_price']):
                self._execute_entry(trigger_data)
                del self.pending_triggers[symbol]

        # --- 2. Check for EXIT Conditions for open positions ---
        position = self.portfolio_manager.get_position(symbol)
        if not position: return
        
        exit_reason, exit_price = None, None

        # --- 2a. Breakeven Logic ---
        if not position['be_activated']:
            current_r = 0
            if position['direction'] == 'LONG':
                current_r = (price - position['entry_price']) / position['initial_risk_per_share']
            else: # SHORT
                current_r = (position['entry_price'] - price) / position['initial_risk_per_share']
            
            if current_r >= config.BREAKEVEN_TRIGGER_R:
                if position['direction'] == 'LONG':
                    new_sl = position['entry_price'] + (position['initial_risk_per_share'] * config.BREAKEVEN_PROFIT_R)
                else: # SHORT
                    new_sl = position['entry_price'] - (position['initial_risk_per_share'] * config.BREAKEVEN_PROFIT_R)
                position['sl'] = new_sl
                position['be_activated'] = True
                self.logger.log_console("UPDATE", f"[{position['direction']}] {symbol} - Breakeven activated. New SL: {new_sl:.2f}")

        # --- 2b. Check standard SL/TP hits ---
        if position['direction'] == 'LONG':
            if price <= position['sl']: exit_reason, exit_price = 'SL_HIT', position['sl']
            elif price >= position['tp']: exit_reason, exit_price = 'TP_HIT', position['tp']
        else: # SHORT
            if price >= position['sl']: exit_reason, exit_price = 'SL_HIT', position['sl']
            elif price <= position['tp']: exit_reason, exit_price = 'TP_HIT', position['tp']
        
        if exit_reason:
            self.portfolio_manager.close_position(symbol, exit_price, exit_reason)

    def on_15min_candle_for_tsl(self, data):
        """Listens for candle closes to update the ATR trailing stop loss."""
        symbol = data['symbol']
        atr = data.get('atr')
        
        position = self.portfolio_manager.get_position(symbol)
        if not position or not atr: return

        # Determine current profit in R to check for aggressive TSL
        last_close = data['candle']['close']
        current_r = 0
        if position['direction'] == 'LONG':
            current_r = (last_close - position['entry_price']) / position['initial_risk_per_share']
        else: # SHORT
            current_r = (position['entry_price'] - last_close) / position['initial_risk_per_share']

        # Determine which multiplier to use
        is_aggressive = (position['direction'] == 'LONG' and current_r >= config.AGGRESSIVE_TS_TRIGGER_R_LONG) or \
                        (position['direction'] == 'SHORT' and current_r >= config.AGGRESSIVE_TS_TRIGGER_R_SHORT)
        
        if is_aggressive:
            position['current_ts_multiplier'] = config.AGGRESSIVE_TS_MULTIPLIER
        
        # Update SL based on ATR TSL logic
        new_sl = 0
        if position['direction'] == 'LONG':
            new_sl = last_close - (atr * position['current_ts_multiplier'])
            # TSL can only move up, not down
            if new_sl > position['sl']:
                position['sl'] = new_sl
                self.logger.log_console("UPDATE", f"[LONG] {symbol} - TSL updated to {new_sl:.2f} (Aggressive: {is_aggressive})")
        else: # SHORT
            new_sl = last_close + (atr * position['current_ts_multiplier'])
            # TSL can only move down, not up
            if new_sl < position['sl']:
                position['sl'] = new_sl
                self.logger.log_console("UPDATE", f"[SHORT] {symbol} - TSL updated to {new_sl:.2f} (Aggressive: {is_aggressive})")

    def _execute_entry(self, signal_data):
        """Handles the logic of entering a trade after validating with the portfolio manager."""
        # Simulate latency
        time.sleep(random.uniform(config.SIMULATED_LATENCY_MS[0], config.SIMULATED_LATENCY_MS[1]) / 1000.0)

        # Simulate slippage
        # NOTE: A proper implementation would need the instrument's tick size. We'll use a small percentage for now.
        trigger_price = signal_data['trigger_price']
        slippage = trigger_price * (config.SLIPPAGE_FACTOR / 10000) # Simple percentage based slippage
        
        if signal_data['direction'] == 'LONG':
            fill_price = trigger_price + slippage
        else: # SHORT
            fill_price = trigger_price - slippage
            
        self.portfolio_manager.open_position(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=fill_price,
            sl_price=signal_data['sl_price']
        )

