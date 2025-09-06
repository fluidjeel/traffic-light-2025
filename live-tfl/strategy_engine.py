# ------------------------------------------------------------------------------------------------
# strategy_engine.py - The Brains of the TFL Trading Bot
# ------------------------------------------------------------------------------------------------
#
# ARCHITECTURAL FIX: This component is now stateless and purely reactive.
# - It no longer stores or modifies any candle data.
# - It subscribes to 'CANDLE_CLOSED_15MIN' notifications.
# - On notification, it requests the complete, clean history from the DataHandler.
# - This prevents data corruption and ensures reliable signal generation.
#
# ------------------------------------------------------------------------------------------------

import pandas as pd
import pandas_ta as ta
from datetime import time, datetime
import warnings

import config

class StrategyEngine:
    def __init__(self, symbols, event_bus, logger, data_handler):
        self.symbols = symbols
        self.event_bus = event_bus
        self.logger = logger
        self.data_handler = data_handler

        self.daily_data = {} # {symbol: {'rsi': val}}
        
        self.event_bus.subscribe('CANDLE_CLOSED_15MIN', self.on_15min_candle)
        self._pre_calculate_daily_indicators()

    def _pre_calculate_daily_indicators(self):
        """Fetches daily data at startup to calculate long-term indicators like RSI."""
        self.logger.log_console("INFO", "Strategy Engine initializing: Calculating daily indicators...")
        for symbol in self.symbols:
            daily_df = self.data_handler.get_historical_data(symbol, resolution='D', days_of_data=200)
            if not daily_df.empty and len(daily_df) > config.DAILY_RSI_PERIOD:
                try:
                    daily_df['rsi'] = ta.rsi(daily_df['close'], length=config.DAILY_RSI_PERIOD)
                    last_rsi = daily_df['rsi'].iloc[-1]
                    self.daily_data[symbol] = {'rsi': last_rsi}
                    self.logger.log_to_csv('data_health_log', {'timestamp': datetime.now(), 'event_type': 'INDICATOR_CALC', 'symbol': symbol, 'status': 'SUCCESS', 'details': f'Daily RSI calculated: {last_rsi:.2f}'})
                except Exception as e:
                    self.logger.log_to_csv('data_health_log', {'timestamp': datetime.now(), 'event_type': 'INDICATOR_CALC', 'symbol': symbol, 'status': 'FAIL', 'details': f'RSI calculation error: {e}'})
            else:
                self.logger.log_to_csv('data_health_log', {'timestamp': datetime.now(), 'event_type': 'INDICATOR_CALC', 'symbol': symbol, 'status': 'FAIL', 'details': 'Insufficient daily data for RSI.'})

    def run_sanity_check(self):
        """Performs a 'dry run' using the pre-loaded historical data."""
        self.logger.log_console("INFO", "--- Running Startup Sanity Check ---")
        for symbol in self.symbols:
            try:
                # --- FIX: Use the already loaded historical data from DataHandler ---
                hist_15min_df = self.data_handler.candles_15min.get(symbol)
                
                if hist_15min_df is None or hist_15min_df.empty or len(hist_15min_df) < config.INTRADAY_MVWAP_PERIOD:
                    self.logger.log_console("WARN", f"[Sanity Check] Insufficient historical 15m data for {symbol}.")
                    continue
                
                history_copy = hist_15min_df.copy()
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    history_copy['vwap'] = ta.vwap(high=history_copy['high'], low=history_copy['low'], close=history_copy['close'], volume=history_copy['volume'], length=config.INTRADAY_MVWAP_PERIOD)
                
                history_copy['atr'] = ta.atr(high=history_copy['high'], low=history_copy['low'], close=history_copy['close'], length=config.ATR_TS_PERIOD)
                
                self.check_tfl_pattern(symbol, history_copy)

            except Exception as e:
                self.logger.log_console("ERROR", f"[Sanity Check] Failed for {symbol}: {e}")
        self.logger.log_console("INFO", "--- Startup Sanity Check Complete ---")


    def on_15min_candle(self, data):
        """The main event handler for incoming 15-minute candles."""
        try:
            symbol = data['symbol']
            
            # --- FIX: Request clean, complete data from the DataHandler ---
            candle_history_df = self.data_handler.candles_15min.get(symbol)
            if candle_history_df is None: return

            # --- FIX: Check timing rules based on the LAST candle in the history ---
            if config.AVOID_OPEN_CLOSE_ENTRIES:
                candle_time = candle_history_df.index[-1].time()
                if candle_time == time(9, 15) or candle_time == time(15, 15):
                    return

            min_data_length = max(config.MAX_RED_CANDLES + 2, config.INTRADAY_MVWAP_PERIOD, config.ATR_TS_PERIOD)
            if len(candle_history_df) < min_data_length:
                return

            history_copy = candle_history_df.copy()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                history_copy['vwap'] = ta.vwap(high=history_copy['high'], low=history_copy['low'], close=history_copy['close'], volume=history_copy['volume'], length=config.INTRADAY_MVWAP_PERIOD)

            history_copy['atr'] = ta.atr(high=history_copy['high'], low=history_copy['low'], close=history_copy['close'], length=config.ATR_TS_PERIOD)
            
            # --- FIX: Publish the ATR update on the correct event channel ---
            self.event_bus.publish('ATR_UPDATE', {
                'symbol': symbol,
                'atr': history_copy['atr'].iloc[-1]
            })

            self.check_tfl_pattern(symbol, history_copy)

        except Exception as e:
            self.logger.log_console("ERROR", f"StrategyEngine error in on_15min_candle for {data.get('symbol', 'UNKNOWN')}: {e}")

    def check_tfl_pattern(self, symbol, df):
        """Checks for the TFL price action pattern and applies all filters."""
        if df.empty: return
        
        last_candle = df.iloc[-1]
        daily_rsi = self.daily_data.get(symbol, {}).get('rsi', 50)
        last_vwap = df['vwap'].iloc[-1] if 'vwap' in df.columns and not pd.isna(df['vwap'].iloc[-1]) else 0
        
        df_for_pattern = df.iloc[:-1] 
        long_pattern, pattern_high, pattern_low = self._find_long_pattern(df_for_pattern)
        short_pattern, pattern_high_s, pattern_low_s = self._find_short_pattern(df_for_pattern)
        
        log_data = {
            'timestamp': df.index[-1], 'symbol': symbol, 'price': last_candle['close'], 'daily_rsi': f"{daily_rsi:.2f}",
            'is_rsi_ok': 'N/A', 'mvwap': f"{last_vwap:.2f}", 'is_mvwap_ok': 'N/A',
            'pattern_found': 'NO_PATTERN', 'is_setup_valid': False, 'rejection_reason': ''
        }
        
        if long_pattern:
            is_long_rsi_ok = daily_rsi > config.DAILY_RSI_LONG_THRESHOLD
            is_long_mvwap_ok = df_for_pattern.iloc[-1]['close'] > df_for_pattern.iloc[-1]['vwap']
            
            log_data.update({'pattern_found': 'TFL_LONG', 'is_rsi_ok': is_long_rsi_ok, 'is_mvwap_ok': is_long_mvwap_ok})
            
            if is_long_rsi_ok and is_long_mvwap_ok:
                log_data['is_setup_valid'] = True
                self.logger.log_console("SETUP", f"[LONG] {symbol} - TFL pattern found. RSI={daily_rsi:.1f}. Awaiting breakout of {pattern_high:.2f}")
                self.event_bus.publish('PotentialTradeSignal', {
                    'symbol': symbol, 'direction': 'LONG',
                    'trigger_price': pattern_high, 'sl_price': pattern_low
                })
            else:
                reasons = []
                if not is_long_rsi_ok: reasons.append("RSI_FILTER_FAILED")
                if not is_long_mvwap_ok: reasons.append("MVWAP_FILTER_FAILED")
                log_data['rejection_reason'] = "&".join(reasons)

        elif short_pattern:
            is_short_rsi_ok = daily_rsi < config.DAILY_RSI_SHORT_THRESHOLD
            is_short_mvwap_ok = df_for_pattern.iloc[-1]['close'] < df_for_pattern.iloc[-1]['vwap']

            log_data.update({'pattern_found': 'TFL_SHORT', 'is_rsi_ok': is_short_rsi_ok, 'is_mvwap_ok': is_short_mvwap_ok})

            if is_short_rsi_ok and is_short_mvwap_ok:
                log_data['is_setup_valid'] = True
                self.logger.log_console("SETUP", f"[SHORT] {symbol} - TFL pattern found. RSI={daily_rsi:.1f}. Awaiting breakdown of {pattern_low_s:.2f}")
                self.event_bus.publish('PotentialTradeSignal', {
                    'symbol': symbol, 'direction': 'SHORT',
                    'trigger_price': pattern_low_s, 'sl_price': pattern_high_s
                })
            else:
                reasons = []
                if not is_short_rsi_ok: reasons.append("RSI_FILTER_FAILED")
                if not is_short_mvwap_ok: reasons.append("MVWAP_FILTER_FAILED")
                log_data['rejection_reason'] = "&".join(reasons)

        self.logger.log_to_csv('scanner_log', log_data)

    def _find_long_pattern(self, df):
        """Iterates backwards to find the TFL long pattern (pullback of red candles, then one green)."""
        if len(df) < 2: return False, None, None
        
        if df.iloc[-1]['close'] <= df.iloc[-1]['open']: return False, None, None
        
        setup_candle = df.iloc[-1]
        pattern_high = setup_candle['high']
        pattern_low = setup_candle['low']
        
        red_candle_count = 0
        for i in range(2, config.MAX_RED_CANDLES + 3):
            if i > len(df): break
            
            prev_candle = df.iloc[-i]
            if prev_candle['close'] < prev_candle['open']:
                red_candle_count += 1
                pattern_high = max(pattern_high, prev_candle['high'])
                pattern_low = min(pattern_low, prev_candle['low'])
            else:
                break
        
        if config.MIN_RED_CANDLES <= red_candle_count <= config.MAX_RED_CANDLES:
            return True, pattern_high, pattern_low
        
        return False, None, None

    def _find_short_pattern(self, df):
        """Iterates backwards to find the TFL short pattern (pullback of green candles, then one red)."""
        if len(df) < 2: return False, None, None

        if df.iloc[-1]['close'] >= df.iloc[-1]['open']: return False, None, None
        
        setup_candle = df.iloc[-1]
        pattern_high = setup_candle['high']
        pattern_low = setup_candle['low']
        
        green_candle_count = 0
        for i in range(2, config.MAX_RED_CANDLES + 3):
            if i > len(df): break
            
            prev_candle = df.iloc[-i]
            if prev_candle['close'] > prev_candle['open']:
                green_candle_count += 1
                pattern_high = max(pattern_high, prev_candle['high'])
                pattern_low = min(pattern_low, prev_candle['low'])
            else:
                break
        
        if config.MIN_RED_CANDLES <= green_candle_count <= config.MAX_RED_CANDLES:
            return True, pattern_high, pattern_low
            
        return False, None, None

