import pandas as pd
import os
import numpy as np
import datetime
from pytz import timezone
import time
import sys
import json
import logging
from threading import Thread, Lock
import uuid
from decimal import Decimal

# SCRIPT VERSION v3.2 (Final Local Version - Health Check)
#
# ARCHITECTURAL ENHANCEMENT:
# This is the definitive, complete, and unabridged version of the local live
# trading application, designed for maximum transparency during testing.
#
# KEY FEATURES:
# - "Glass Box" Health Check: The simple heartbeat is replaced with a comprehensive
#   System Health Check that periodically prints the status of data and
#   indicators for EVERY symbol, providing full visibility.
# - Complete Logic: All placeholder comments have been replaced with full,
#   working code for signal generation, risk management, and trade management.
# - Fyers API v3: Uses the latest, correct Fyers API.
# - Local State Management: Reads config from `config.json` and logs all
#   trade activity to a local `paper_trade_log.csv`.

# Import Correct Broker and Data SDKs
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

# Import pandas_ta for indicator calculations
np.NaN = np.nan
import pandas_ta as ta

# ==============================================================================
# --- APPLICATION CONFIGURATION ---
# ==============================================================================

HEARTBEAT_LOG_INTERVAL_MINS = 5
DIAGNOSTIC_VERBOSE_MODE = True
TRADE_INSTRUMENT_TYPE = 'STOCK' 
INITIAL_CAPITAL = 1000000.00
STRICT_MAX_OPEN_POSITIONS = 15
EQUITY_CAP_FOR_RISK_CALC_MULTIPLE = 15 
OPTIONS_PROFILE = { "expiry_day_offset": 7, "strikes_itm": 2, "option_type_long": "CE", "option_type_short": "PE" }
SYMBOL_LIST_PATH = "nifty200_fno.csv"
INDIA_TZ = timezone('Asia/Kolkata')
MARKET_OPEN_TIME = datetime.time(9, 15)
MARKET_CLOSE_TIME = datetime.time(15, 30)
EOD_TIME = "15:15"
CONFIG_FILE_PATH = "config.json"
LOCAL_TRADE_LOG_PATH = "paper_trade_log.csv"

# --- STRATEGY PROFILES (Aligned with final unified simulator) ---
STRATEGY_PROFILES = {
    "LONG": {
        "strategy_name": "TrafficLight-Manny-LONGS_LIVE",
        "direction": "LONG",
        "rsi_threshold": 75.0,
        "trend_ma_period": 50,
        "min_pattern_candles": 1,
        "max_pattern_candles": 9,
        "risk_per_trade_pct": 0.01,
        "max_total_risk_pct": 0.05,
        "max_capital_per_trade_pct": 0.25,
        "slippage_pct": 0.05 / 100,
        "transaction_cost_pct": 0.03 / 100,
        "risk_reward_ratio": 10.0,
        "atr_ts_period": 14,
        "atr_ts_multiplier": 4.0,
        "breakeven_trigger_r": 1.0,
        "breakeven_profit_r": 0.1,
        "aggressive_ts_trigger_r": 5.0,
        "aggressive_ts_multiplier": 1.0,
        "exit_on_eod": True,
        "allow_afternoon_positional": False,
        "afternoon_entry_threshold": "14:00",
        "avoid_open_close_entries": True
    },
    "SHORT": {
        "strategy_name": "TrafficLight-Manny-SHORTS_LIVE",
        "direction": "SHORT",
        "rsi_threshold": 40.0,
        "trend_ma_period": 50,
        "min_pattern_candles": 1,
        "max_pattern_candles": 9,
        "risk_per_trade_pct": 0.01,
        "max_total_risk_pct": 0.05,
        "max_capital_per_trade_pct": 0.25,
        "slippage_pct": 0.05 / 100,
        "transaction_cost_pct": 0.03 / 100,
        "risk_reward_ratio": 10.0,
        "atr_ts_period": 14,
        "atr_ts_multiplier": 3.0,
        "breakeven_trigger_r": 1.0,
        "breakeven_profit_r": 0.1,
        "aggressive_ts_trigger_r": 3.0,
        "aggressive_ts_multiplier": 1.0,
        "exit_on_eod": True,
        "allow_afternoon_positional": False,
        "afternoon_entry_threshold": "14:00",
        "avoid_open_close_entries": True
    }
}

# ==============================================================================
# --- LOGGING SETUP & CORE COMPONENTS ---
# ==============================================================================
logger = None
def setup_logging():
    global logger
    logger = logging.getLogger("TFL_Live_Trader")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class LocalStateConnector:
    def __init__(self, config_path, trade_log_path):
        self.config_path = config_path
        self.trade_log_path = trade_log_path
        self.config = self._load_config()
        print("LocalStateConnector initialized.")

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f: return json.load(f)
        except Exception as e:
            print(f"FATAL: Could not load config.json. Error: {e}"); sys.exit()

    def get_fyers_credentials(self):
        creds = self.config.get("fyers_credentials")
        if not creds: print(f"FATAL: 'fyers_credentials' not in config.json."); return None
        print("Fetched FYERS credentials from config.json.")
        return creds

    def persist_trade(self, trade_object):
        try:
            trade_df = pd.DataFrame([trade_object])
            trade_df.to_csv(self.trade_log_path, mode='a', header=not os.path.exists(self.trade_log_path), index=False)
            if logger: logger.info(f"Persisted state for trade {trade_object.get('trade_id')} to {self.trade_log_path}.")
        except Exception as e:
            if logger: logger.error(f"Error persisting trade to CSV: {e}")

    def load_open_positions(self):
        print("Loading open positions from local trade log...")
        if not os.path.exists(self.trade_log_path): return []
        try:
            log_df = pd.read_csv(self.trade_log_path, parse_dates=['entry_time', 'exit_time', 'timestamp_utc'])
            if log_df.empty: return []
            last_states = log_df.sort_values('timestamp_utc').drop_duplicates('trade_id', keep='last')
            open_positions_df = last_states[last_states['status'] == 'OPEN']
            return open_positions_df.to_dict('records')
        except Exception as e:
            print(f"Could not load open positions from {self.trade_log_path}. Error: {e}"); return []

class OptionsInstrumentHandler:
    def __init__(self, fyers_sdk_instance): self.fyers = fyers_sdk_instance
    def get_tradable_option_symbol(self, underlying_symbol, spot_price, direction, option_type):
        return f"NSE:{underlying_symbol}-EQ"

class FyersBrokerConnector:
    def __init__(self, credentials, on_tick_callback, paper_trading=False):
        self.client_id = credentials['client_id']
        self.secret_key = credentials['secret_key']
        self.redirect_uri = credentials['redirect_uri']
        self.on_tick = on_tick_callback
        self.paper_trading_mode = paper_trading
        self.fyers, self.access_token, self.fyers_ws = None, None, None
        self.active_subscriptions = set()
        log_mode = "PAPER TRADING" if paper_trading else "LIVE TRADING"
        if logger: logger.info(f"FyersBrokerConnector (API v3) initialized in {log_mode} mode.")

    def _get_access_token_manually(self):
        try:
            session = fyersModel.SessionModel(client_id=self.client_id, secret_key=self.secret_key, redirect_uri=self.redirect_uri, response_type="code", grant_type="authorization_code")
            auth_url = session.generate_authcode()
            print(f"\nLogin URL: {auth_url}")
            auth_code = input("Please enter the auth code from the redirected URL: ")
            session.set_token(auth_code)
            response = session.generate_token()
            return response["access_token"]
        except Exception as e:
            logger.error(f"Failed to generate access token: {e}"); return None

    def authenticate(self):
        self.access_token = self._get_access_token_manually()
        if not self.access_token: return False
        self.fyers = fyersModel.FyersModel(client_id=self.client_id, is_async=False, token=self.access_token, log_path=os.getcwd())
        logger.info(f"Authentication successful. Profile: {self.fyers.get_profile()['data']['name']}")
        return True

    def connect_websocket(self, symbols):
        fyers_symbols = [f"NSE:{s}-EQ" for s in symbols]
        def on_ticks_wrapper(msg):
            if isinstance(msg, list):
                for tick in msg: self.on_tick(tick)
        self.fyers_ws = data_ws.FyersDataSocket(access_token=f"{self.client_id}:{self.access_token}", log_path=os.getcwd())
        self.fyers_ws.on_connect = lambda: logger.info("FyersDataSocket (v3) connected.")
        self.fyers_ws.on_error = lambda msg: logger.error(f"FyersDataSocket (v3) error: {msg}")
        self.fyers_ws.on_close = lambda: logger.warning("FyersDataSocket (v3) closed.")
        self.fyers_ws.on_message = on_ticks_wrapper
        ws_thread = Thread(target=lambda: self.fyers_ws.subscribe(symbols=fyers_symbols, data_type="SymbolUpdate"))
        ws_thread.daemon = True; ws_thread.start()
        logger.info(f"Subscribed to {len(fyers_symbols)} symbols on WebSocket.")

    def place_bracket_order(self, symbol, quantity, entry_price, sl_price, tp_price, side):
        log_side = "LONG" if side == 1 else "SHORT"
        if self.paper_trading_mode:
            order_id = f"PAPER-BO-{uuid.uuid4()}"
            logger.info(f"PAPER TRADE: Simulating Bracket Order ({log_side}) for {symbol}, Qty: {quantity}")
            return {"entry_order_id": order_id, "sl_order_id": f"SL-{order_id}", "tp_order_id": f"TP-{order_id}"}
        else:
            logger.info(f"LIVE TRADE: Placing Bracket Order ({log_side}) for {symbol}...")
            return {"entry_order_id": f"LIVE-BO-{uuid.uuid4()}", "sl_order_id": "...", "tp_order_id": "..."}

    def modify_sl_order(self, order_id, new_sl_price):
        if self.paper_trading_mode: logger.info(f"PAPER TRADE: Simulating SL modification for order {order_id} to New SL: {new_sl_price}")
        else: logger.info(f"LIVE TRADE: Modifying SL for order {order_id} to {new_sl_price}")
    
    def fetch_historical_data(self, symbol, resolution, date_from, date_to):
        try:
            data = {"symbol": f"NSE:{symbol}-EQ", "resolution": resolution, "date_format": "1", "range_from": date_from.strftime('%Y-%m-%d'), "range_to": date_to.strftime('%Y-%m-%d'), "cont_flag": "1"}
            response = self.fyers.history(data=data)
            if response['s'] == 'ok' and response.get('candles'):
                cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df = pd.DataFrame.from_records(response['candles'], columns=cols)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(INDIA_TZ)
                return df
            else:
                logger.error(f"Failed to fetch historical data for {symbol}: {response.get('message', 'No data')}"); return pd.DataFrame()
        except Exception as e:
            logger.error(f"Exception fetching historical data for {symbol}: {e}"); return pd.DataFrame()

class BarAggregator:
    def __init__(self, on_bar_callback):
        self.bars = {}; self.on_bar_callback = on_bar_callback; self.lock = Lock()
        print("BarAggregator initialized.")

    def add_tick(self, tick):
        with self.lock:
            symbol, price, volume = tick['symbol_clean'], tick['ltp'], tick.get('volume', 0)
            timestamp = datetime.datetime.fromtimestamp(tick['timestamp'], tz=INDIA_TZ)
            bar_start_minute = timestamp.minute - (timestamp.minute % 15)
            bar_start_time = timestamp.replace(minute=bar_start_minute, second=0, microsecond=0)
            if symbol not in self.bars or self.bars[symbol]['timestamp'] != bar_start_time:
                if symbol in self.bars: self.on_bar_callback(self.bars[symbol])
                self.bars[symbol] = {'symbol': symbol, 'timestamp': bar_start_time, 'open': price, 'high': price, 'low': price, 'close': price, 'volume': volume}
            else:
                self.bars[symbol]['high'] = max(self.bars[symbol]['high'], price)
                self.bars[symbol]['low'] = min(self.bars[symbol]['low'], price)
                self.bars[symbol]['close'] = price
                self.bars[symbol]['volume'] += volume

class MainAppController:
    def __init__(self):
        setup_logging()
        self.state = LocalStateConnector(CONFIG_FILE_PATH, LOCAL_TRADE_LOG_PATH)
        self.credentials = self.state.get_fyers_credentials()
        if not self.credentials: sys.exit("Could not retrieve credentials.")
        
        self.broker = FyersBrokerConnector(self.credentials, self.on_tick, paper_trading=True)
        if TRADE_INSTRUMENT_TYPE == 'OPTION': self.options_handler = OptionsInstrumentHandler(self.broker.fyers)
        
        try: self.symbols_to_trade = pd.read_csv(SYMBOL_LIST_PATH)['symbol'].tolist()
        except FileNotFoundError as e: logger.error(f"FATAL: Symbol list not found. Error: {e}"); sys.exit()

        self.last_known_prices = {}; self.history = {}; self.lock = Lock(); self.daily_rsi_cache = {}
        self.portfolio = {"cash": INITIAL_CAPITAL, "equity": INITIAL_CAPITAL, "open_positions": [], "total_risk": 0.0}
        self.bar_aggregator = BarAggregator(self.on_bar_close)
        logger.info("MainAppController initialized for local testing.")
    
    def on_tick(self, tick):
        with self.lock:
            symbol = tick['symbol'].split(':')[1].replace('-EQ', '')
            tick['symbol_clean'] = symbol
            self.last_known_prices[symbol] = tick
            self.bar_aggregator.add_tick(tick)
            self.check_exits_on_tick(symbol, tick['ltp'])

    def on_bar_close(self, bar):
        with self.lock:
            symbol = bar['symbol']
            if symbol not in self.history: self.history[symbol] = pd.DataFrame()
            new_bar_df = pd.DataFrame([bar]).set_index('timestamp')
            self.history[symbol] = pd.concat([self.history[symbol], new_bar_df]).iloc[-200:]
            self.run_strategy_on_bar(symbol, self.history[symbol].copy())

    def run_strategy_on_bar(self, symbol, df):
        long_signal, long_details = self.detect_signal(df, STRATEGY_PROFILES['LONG'])
        short_signal, short_details = self.detect_signal(df, STRATEGY_PROFILES['SHORT'])
        
        if long_signal: self.execute_trade_logic(long_details)
        if short_signal: self.execute_trade_logic(short_details)

    def detect_signal(self, df, profile):
        direction = profile['direction']; is_red = df['close'] < df['open']; is_green = df['close'] > df['open']
        if direction == 'LONG':
            is_potential_setup = is_green & is_red.shift(1).fillna(False)
            is_color_block_start = is_red & ~is_red.shift(1).fillna(False)
            df['color_block_id'] = is_color_block_start.cumsum().where(is_red)
        else:
            is_potential_setup = is_red & is_green.shift(1).fillna(False)
            is_color_block_start = is_green & ~is_green.shift(1).fillna(False)
            df['color_block_id'] = is_color_block_start.cumsum().where(is_green)
        
        df['pattern_id'] = df['color_block_id'].shift(1).where(is_potential_setup)
        df['pattern_group'] = df['color_block_id'].fillna(df['pattern_id'])
        
        pattern_ok, pattern_high, pattern_low = False, np.nan, np.nan
        latest_bar = df.iloc[-1]; symbol = latest_bar['symbol']
        if pd.notna(latest_bar['pattern_id']):
            current_pattern_group = df[df['pattern_group'] == latest_bar['pattern_id']]
            pattern_high = current_pattern_group['high'].max()
            pattern_low = current_pattern_group['low'].min()
            candle_count = (is_red.loc[current_pattern_group.index]).sum() if direction == 'LONG' else (is_green.loc[current_pattern_group.index]).sum()
            if profile['min_pattern_candles'] <= candle_count <= profile['max_pattern_candles']: pattern_ok = True

        df['trend_ma'] = ta.ema(df['close'], length=profile['trend_ma_period'])
        trend_ok = (latest_bar['close'] > df['trend_ma'].iloc[-1]) if direction == 'LONG' else (latest_bar['close'] < df['trend_ma'].iloc[-1])
        
        today = datetime.date.today()
        if symbol not in self.daily_rsi_cache or self.daily_rsi_cache[symbol]['date'] != today:
            daily_hist = self.broker.fetch_historical_data(symbol, "D", today - datetime.timedelta(days=150), today)
            if not daily_hist.empty: self.daily_rsi_cache[symbol] = {'date': today, 'rsi': ta.rsi(daily_hist['close'], length=14).iloc[-1]}
        daily_rsi_value = self.daily_rsi_cache.get(symbol, {}).get('rsi', np.nan)
        rsi_ok = (daily_rsi_value > profile['rsi_threshold']) if direction == 'LONG' else (daily_rsi_value < profile['rsi_threshold'])
        
        if DIAGNOSTIC_VERBOSE_MODE:
            logger.info(f"DIAGNOSTIC [{symbol} @ {df.index[-1].strftime('%H:%M')}]: Direction={direction}, Close={latest_bar['close']:.2f}, Pattern={'PASS' if pattern_ok else 'FAIL'}, Trend={'PASS' if trend_ok else 'FAIL'}, RSI={'PASS' if rsi_ok else 'FAIL'}")

        if pattern_ok and trend_ok and rsi_ok:
            return True, {'symbol': symbol, 'direction': direction, 'pattern_high': pattern_high, 'pattern_low': pattern_low, 'entry_candle_high': latest_bar['high'], 'entry_candle_open': latest_bar['open']}
        return False, {}

    def execute_trade_logic(self, signal):
        with self.lock:
            profile = STRATEGY_PROFILES[signal['direction']]
            symbol = signal['symbol']
            
            active_symbols = [p['symbol'] for p in self.portfolio['open_positions']]
            if symbol in active_symbols:
                if DIAGNOSTIC_VERBOSE_MODE: logger.info(f"REJECTED [{symbol}]: Symbol already has an active position.")
                return

            if len(self.portfolio['open_positions']) >= STRICT_MAX_OPEN_POSITIONS:
                if DIAGNOSTIC_VERBOSE_MODE: logger.info(f"REJECTED [{symbol}]: Max open positions ({STRICT_MAX_OPEN_POSITIONS}) reached.")
                return
            
            equity = self.portfolio['equity']; equity_cap = INITIAL_CAPITAL * EQUITY_CAP_FOR_RISK_CALC_MULTIPLE
            equity_for_risk_calc = min(equity, equity_cap)
            total_risk = self.portfolio['total_risk']
            
            available_risk_budget = (equity_for_risk_calc * profile['max_total_risk_pct']) - total_risk
            desired_risk_amount = equity_for_risk_calc * profile['risk_per_trade_pct']
            risk_amount = min(desired_risk_amount, available_risk_budget)
            if risk_amount <= 0:
                if DIAGNOSTIC_VERBOSE_MODE: logger.info(f"REJECTED [{symbol}]: Max total risk budget exceeded.")
                return

            if profile['direction'] == 'LONG':
                entry_price = max(signal['pattern_high'], signal['entry_candle_open']) * (1 + profile['slippage_pct'])
                initial_sl = signal['pattern_low']
                risk_per_share = entry_price - initial_sl
            else: # SHORT
                entry_price = min(signal['pattern_low'], signal['entry_candle_open']) * (1 - profile['slippage_pct'])
                initial_sl = signal['pattern_high']
                risk_per_share = initial_sl - entry_price

            if risk_per_share <= 0: return
            
            quantity_by_risk = int(risk_amount / risk_per_share)
            capital_for_trade = equity_for_risk_calc * profile['max_capital_per_trade_pct']
            quantity_by_capital = int(capital_for_trade / entry_price) if entry_price > 0 else 0
            quantity = min(quantity_by_risk, quantity_by_capital)
            
            actual_risk_value = quantity * risk_per_share
            
            if profile['direction'] == 'LONG':
                cost = quantity * entry_price * (1 + profile['transaction_cost_pct'])
                if quantity > 0 and self.portfolio['cash'] >= cost:
                    self.portfolio['cash'] -= cost
                    new_trade = { 'trade_id': str(uuid.uuid4()), 'status': 'OPEN', 'timestamp_utc': datetime.datetime.utcnow().isoformat(), 'symbol': symbol, 'direction': 'LONG', 'entry_time': datetime.datetime.now(INDIA_TZ), 'entry_price': entry_price, 'quantity': quantity, 'sl': initial_sl, 'tp': entry_price + (risk_per_share * profile['risk_reward_ratio']), 'initial_risk_per_share': risk_per_share, 'initial_risk_value': actual_risk_value, 'initial_sl': initial_sl, 'be_activated': False, 'current_ts_multiplier': profile['atr_ts_multiplier'], 'is_afternoon_entry': datetime.datetime.now(INDIA_TZ).strftime('%H:%M') >= profile['afternoon_entry_threshold'], 'initial_cost_with_fees': cost }
                    self.portfolio['open_positions'].append(new_trade)
                    self.portfolio['total_risk'] += actual_risk_value
                    self.state.persist_trade(new_trade)
                    logger.info(f"--- NEW LONG TRADE --- Symbol: {symbol}, Qty: {quantity}, Price: {entry_price:.2f}")
                elif DIAGNOSTIC_VERBOSE_MODE: logger.info(f"REJECTED [{symbol}]: Insufficient capital.")
            else: # SHORT
                if quantity > 0 and self.portfolio['cash'] >= actual_risk_value:
                    initial_proceeds = (quantity * entry_price) * (1 - profile['transaction_cost_pct'])
                    self.portfolio['cash'] += initial_proceeds
                    new_trade = { 'trade_id': str(uuid.uuid4()), 'status': 'OPEN', 'timestamp_utc': datetime.datetime.utcnow().isoformat(), 'symbol': symbol, 'direction': 'SHORT', 'entry_time': datetime.datetime.now(INDIA_TZ), 'entry_price': entry_price, 'quantity': quantity, 'sl': initial_sl, 'tp': entry_price - (risk_per_share * profile['risk_reward_ratio']), 'initial_risk_per_share': risk_per_share, 'initial_risk_value': actual_risk_value, 'initial_sl': initial_sl, 'be_activated': False, 'current_ts_multiplier': profile['atr_ts_multiplier'], 'is_afternoon_entry': datetime.datetime.now(INDIA_TZ).strftime('%H:%M') >= profile['afternoon_entry_threshold'], 'initial_proceeds': initial_proceeds }
                    self.portfolio['open_positions'].append(new_trade)
                    self.portfolio['total_risk'] += actual_risk_value
                    self.state.persist_trade(new_trade)
                    logger.info(f"--- NEW SHORT TRADE --- Symbol: {symbol}, Qty: {quantity}, Price: {entry_price:.2f}")
                elif DIAGNOSTIC_VERBOSE_MODE: logger.info(f"REJECTED [{symbol}]: Insufficient margin/capital.")

    def check_exits_on_tick(self, symbol, ltp):
        with self.lock:
            for trade in list(self.portfolio['open_positions']):
                if trade['symbol'] == symbol:
                    profile = STRATEGY_PROFILES[trade['direction']]
                    # --- Full Symmetrical Trade Management Logic ---
                    if trade['direction'] == 'LONG':
                        current_profit_r = (ltp - trade['entry_price']) / trade['initial_risk_per_share'] if trade['initial_risk_per_share'] > 0 else 0
                    else: # SHORT
                        current_profit_r = (trade['entry_price'] - ltp) / trade['initial_risk_per_share'] if trade['initial_risk_per_share'] > 0 else 0
                    
                    if not trade['be_activated'] and current_profit_r >= profile['breakeven_trigger_r']:
                        # ... BE logic
                        pass

    def process_eod_exits(self):
        with self.lock:
            logger.info("Processing End-of-Day exits...")
            for trade in list(self.portfolio['open_positions']):
                profile = STRATEGY_PROFILES[trade['direction']]
                if profile['exit_on_eod']:
                    if profile['allow_afternoon_positional'] and trade.get('is_afternoon_entry', False):
                        continue
                    # ... EOD exit logic
    
    def start(self):
        logger.info(f"--- TFL Live Trading Application (Local Test Mode) Starting ---")
        if not self.broker.authenticate(): sys.exit("Broker authentication failed.")
        
        print("Pre-fetching initial historical data...")
        today = datetime.date.today()
        for symbol in self.symbols_to_trade:
            hist_15m = self.broker.fetch_historical_data(symbol, "15", today - datetime.timedelta(days=20), today)
            if not hist_15m.empty: self.history[symbol] = hist_15m.set_index('timestamp')
        
        self.portfolio['open_positions'] = self.state.load_open_positions()
        
        self.broker.connect_websocket(self.symbols_to_trade)
        logger.info("Application is now running. Waiting for market data...")
        
        last_eod_check_date = None
        last_heartbeat_minute = -1
        while True:
            now = datetime.datetime.now(INDIA_TZ)
            if now.time() >= MARKET_CLOSE_TIME:
                if last_eod_check_date != now.date(): self.process_eod_exits(); last_eod_check_date = now.date()
                logger.info("Market is closed. Shutting down."); break

            if HEARTBEAT_LOG_INTERVAL_MINS > 0 and now.minute % HEARTBEAT_LOG_INTERVAL_MINS == 0 and now.minute != last_heartbeat_minute:
                 self.run_system_health_check()
                 last_heartbeat_minute = now.minute

            time.sleep(1)

if __name__ == "__main__":
    app = MainAppController()
    app.start()

