# ------------------------------------------------------------------------------------------------
# strategy_engine.py - The Core TFL Strategy Logic
# ------------------------------------------------------------------------------------------------
#
# This component is the "brain" of the trading bot.
# - It listens for 15-minute candle closure events.
# - When a candle closes, it applies the full TFL strategy rules:
#   1. Daily RSI Filter
#   2. Intraday MVWAP/EMA Filter
#   3. TFL Price Action Pattern Recognition
# - If all conditions are met, it identifies a valid setup and calculates the entry trigger price.
# - It then publishes a 'PotentialTradeSignal' event to the Event Bus for the
#   Execution Simulator to act upon.
#
# ------------------------------------------------------------------------------------------------

import pandas as pd
import pandas_ta as ta

import config

class StrategyEngine:
    # --- FIX: Accepts the data_handler instance during initialization ---
    def __init__(self, symbols, event_bus, logger, data_handler):
        self.symbols = symbols
        self.event_bus = event_bus
        self.logger = logger
        self.data_handler = data_handler # Store the connected data handler instance
        
        # Internal state to hold daily indicator data
        self.daily_indicators = {symbol: {} for symbol in self.symbols}
        
        self.event_bus.subscribe('CANDLE_CLOSED_15MIN', self.on_15min_candle)
        
        self.logger.log_console("INFO", "Strategy Engine initializing: Calculating daily indicators...")
        self._initialize_indicators()

    def _initialize_indicators(self):
        """Fetches daily data and calculates RSI for all symbols at startup."""
        for symbol in self.symbols:
            try:
                # --- FIX: Uses the passed data_handler instance to fetch data ---
                daily_df = self.data_handler.get_historical_data(symbol, resolution='D', days_of_data=100)

                if daily_df.empty or len(daily_df) < config.DAILY_RSI_PERIOD:
                    self.logger.log_to_csv('data_health_log', {
                        'event_type': 'INDICATOR_CALC', 'symbol': symbol, 'status': 'FAIL',
                        'details': 'Insufficient daily data for RSI calculation.'
                    })
                    continue

                # Calculate Daily RSI
                daily_df['rsi'] = ta.rsi(daily_df['close'], length=config.DAILY_RSI_PERIOD)
        # Publish the event WITH the calculated ATR
        self.event_bus.publish('CANDLE_CLOSED_15MIN', {
            'symbol': symbol, 
            'candle': agg_candle,
            'atr': last_atr # Add ATR to the event payload
        })
        self.logger.log_console("DEBUG", f"15m candle closed for {symbol} @ {candle_15min_time.strftime('%H:%M')}")

    def on_15min_candle(self, data):
        """The main event handler for incoming 15-minute candles."""
        symbol = data['symbol']
        candle_history_df = self.data_handler.candles_15min[symbol]
        
        if len(candle_history_df) < max(config.MAX_RED_CANDLES + 2, config.ATR_TS_PERIOD):
            return

        # --- Calculate ATR and add it to the dataframe ---
        candle_history_df['atr'] = ta.atr(
            candle_history_df['high'], 
            candle_history_df['low'], 
            candle_history_df['close'], 
            length=config.ATR_TS_PERIOD
        )
        
        last_atr = candle_history_df['atr'].iloc[-1]
        data['atr'] = last_atr # Add ATR to the data payload for use in this method

        self.check_tfl_pattern(symbol, candle_history_df)

    def check_tfl_pattern(self, symbol, df):
        """Checks for the TFL price action pattern and applies all filters."""
        # For simplicity, we assume the latest candle is at index -1
        # In a real system, you'd ensure the timestamps align perfectly
        last_candle = df.iloc[-1]
        
        # --- 1. Daily Momentum Filter ---
        daily_rsi = self.daily_indicators[symbol].get('daily_rsi', 50) # Default to neutral if not found
        
        is_long_rsi_ok = daily_rsi > config.DAILY_RSI_LONG_THRESHOLD
        is_short_rsi_ok = daily_rsi < config.DAILY_RSI_SHORT_THRESHOLD
        
        # --- 2. Intraday Trend Filter (MVWAP) ---
        # Note: You'd typically use a library like pandas_ta for this
        # This is a simplified calculation for demonstration
        df['mvwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'], length=config.INTRADAY_MVWAP_PERIOD)
        last_mvwap = df['mvwap'].iloc[-1]
        
        is_long_mvwap_ok = last_candle['close'] > last_mvwap
        is_short_mvwap_ok = last_candle['close'] < last_mvwap

        # --- 3. Price Action Pattern Recognition ---
        # This is a complex logic block that needs careful implementation.
        # It involves iterating backwards from the second-to-last candle.
        # For this example, we'll simulate the outcome.
        
        # Placeholder: a more sophisticated function would live here
        long_pattern, pattern_high, pattern_low = self._find_long_pattern(df)
        short_pattern, pattern_high, pattern_low = self._find_short_pattern(df)
        
        log_data = {
            'symbol': symbol, 'price': last_candle['close'], 'daily_rsi': daily_rsi,
            'is_rsi_ok': 'N/A', 'mvwap': last_mvwap, 'is_mvwap_ok': 'N/A',
            'pattern_found': 'NO_PATTERN', 'is_setup_valid': False, 'rejection_reason': ''
        }
        
        # Check for LONG setup
        if long_pattern:
            log_data.update({'pattern_found': 'TFL_LONG', 'is_rsi_ok': is_long_rsi_ok, 'is_mvwap_ok': is_long_mvwap_ok})
            if is_long_rsi_ok and is_long_mvwap_ok:
                log_data['is_setup_valid'] = True
                self.logger.log_console("SETUP", f"[LONG] {symbol} - TFL pattern found. RSI={daily_rsi:.1f}, Price>{last_mvwap:.2f}. Awaiting breakout of {pattern_high}.")
                self.event_bus.publish('PotentialTradeSignal', {
                    'symbol': symbol, 'direction': 'LONG',
                    'trigger_price': pattern_high, 'sl_price': pattern_low
                })
            else:
                reasons = []
                if not is_long_rsi_ok: reasons.append("RSI_FILTER_FAILED")
                if not is_long_mvwap_ok: reasons.append("MVWAP_FILTER_FAILED")
                log_data['rejection_reason'] = "&".join(reasons)

        # Check for SHORT setup
        elif short_pattern:
            log_data.update({'pattern_found': 'TFL_SHORT', 'is_rsi_ok': is_short_rsi_ok, 'is_mvwap_ok': is_short_mvwap_ok})
            if is_short_rsi_ok and is_short_mvwap_ok:
                log_data['is_setup_valid'] = True
                self.logger.log_console("SETUP", f"[SHORT] {symbol} - TFL pattern found. RSI={daily_rsi:.1f}, Price<{last_mvwap:.2f}. Awaiting breakdown of {pattern_low}.")
                self.event_bus.publish('PotentialTradeSignal', {
                    'symbol': symbol, 'direction': 'SHORT',
                    'trigger_price': pattern_low, 'sl_price': pattern_high
                })
            else:
                reasons = []
                if not is_short_rsi_ok: reasons.append("RSI_FILTER_FAILED")
                if not is_short_mvwap_ok: reasons.append("MVWAP_FILTER_FAILED")
                log_data['rejection_reason'] = "&".join(reasons)

        self.logger.log_to_csv('scanner_log', log_data)

    def _find_long_pattern(self, df):
        """Iterates backwards to find the TFL long pattern."""
        # T-1 must be green
        if df.iloc[-2]['close'] <= df.iloc[-2]['open']:
            return False, None, None
        
        setup_candle = df.iloc[-2]
        pattern_high = setup_candle['high']
        pattern_low = setup_candle['low']
        
        # Look for preceding red candles
        red_candle_count = 0
        for i in range(3, config.MAX_RED_CANDLES + 3):
            if i > len(df): break
            
            prev_candle = df.iloc[-i]
            if prev_candle['close'] < prev_candle['open']:
                red_candle_count += 1
                pattern_high = max(pattern_high, prev_candle['high'])
                pattern_low = min(pattern_low, prev_candle['low'])
            else:
                break # End of consecutive red candles
        
        if config.MIN_RED_CANDLES <= red_candle_count <= config.MAX_RED_CANDLES:
            return True, pattern_high, pattern_low
        
        return False, None, None

    def _find_short_pattern(self, df):
        """Iterates backwards to find the TFL short pattern."""
        # T-1 must be red
        if df.iloc[-2]['close'] >= df.iloc[-2]['open']:
            return False, None, None
        
        setup_candle = df.iloc[-2]
        pattern_high = setup_candle['high']
        pattern_low = setup_candle['low']
        
        # Look for preceding green candles
        green_candle_count = 0
        for i in range(3, config.MAX_RED_CANDLES + 3): # Same logic for max pattern length
            if i > len(df): break
            
            prev_candle = df.iloc[-i]
            if prev_candle['close'] > prev_candle['open']:
                green_candle_count += 1
                pattern_high = max(pattern_high, prev_candle['high'])
                pattern_low = min(pattern_low, prev_candle['low'])
            else:
                break # End of consecutive green candles
        
        if config.MIN_RED_CANDLES <= green_candle_count <= config.MAX_RED_CANDLES:
            return True, pattern_high, pattern_low
            
        return False, None, None


