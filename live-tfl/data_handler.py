# ------------------------------------------------------------------------------------------------
# data_handler.py - The Single Source of Truth for Market Data
# ------------------------------------------------------------------------------------------------
#
# ARCHITECTURAL UPGRADE: This version incorporates all learnings from our zero-trust
# diagnostic testing to create a robust and reliable data connection.
#
# KEY FEATURES:
# - Decoupled Connection/Subscription: Waits for a connection to be fully confirmed
#   before sending subscription requests, eliminating race conditions.
# - Brute-Force Subscription: Subscribes to all relevant data types to guarantee
#   receipt of live ticks from the Fyers API v3.
# - Intelligent Message Parsing: The on_message handler can parse different
#   message formats ('sf', 'dp') to correctly extract live price and volume.
#
# ------------------------------------------------------------------------------------------------

import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
import pytz
import threading

import config

class DataHandler:
    def __init__(self, symbols, event_bus, logger):
        self.symbols = symbols
        self.event_bus = event_bus
        self.logger = logger
        self.timezone = pytz.timezone(config.MARKET_TIMEZONE)
        
        self.fyers_rest_client = None
        self.fyers_ws = None
        self.access_token = self._read_token_from_file()

        self.last_tick_prices = {symbol: 0 for symbol in self.symbols}
        self.candles_1min_data = []
        self.candles_15min = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.last_15min_resample_time = None
        
        self.is_connected = False

    # --- WebSocket Callback Methods ---

    def _on_connect(self):
        """Callback executed upon a successful WebSocket connection."""
        self.logger.log_console("SUCCESS", "Fyers WebSocket connection established.")
        self.is_connected = True

    def _on_error(self, msg):
        """Callback to handle WebSocket errors."""
        self.logger.log_console("ERROR", f"WebSocket Error: {msg}")

    def _on_close(self, msg=""):
        """Callback for when the WebSocket connection is closed."""
        self.logger.log_console("INFO", f"WebSocket connection closed. Reason: {msg}")
        self.is_connected = False

    def _on_message(self, msg):
        """
        Processes incoming messages, emits a heartbeat, and parses different
        message types to find live tick data.
        """
        try:
            self.event_bus.publish('DATA_HANDLER_HEARTBEAT', {'timestamp': datetime.now(self.timezone)})
            
            data = json.loads(msg) if isinstance(msg, str) else msg
            if not isinstance(data, dict): return

            symbol, ltp, volume = None, None, None

            if data.get("type") == "sf": # Symbol Feed
                symbol = data.get("symbol")
                ltp = data.get("ltp")
                volume = data.get("last_traded_qty")
            elif data.get("type") == "dp": # Depth Packet
                symbol = data.get("symbol")
                ltp = data.get("ltp")
                volume = data.get("last_traded_qty")

            if symbol and ltp is not None and volume is not None:
                ts = datetime.fromtimestamp(data.get('timestamp', time.time()), tz=self.timezone)
                self.last_tick_prices[symbol] = ltp
                self.event_bus.publish('TICK', {'symbol': symbol, 'price': ltp, 'timestamp': ts})
                self._aggregate_tick(ts, symbol, ltp, volume)

        except Exception as e:
            self.logger.log_console("ERROR", f"Error processing tick message: {e} | Raw: {msg}")


    # --- Connection and Data Handling Logic ---

    def _read_token_from_file(self):
        if os.path.exists(config.TOKEN_FILE):
            with open(config.TOKEN_FILE, 'r') as f: return f.read().strip()
        return None

    def _generate_new_token(self):
        try:
            session = fyersModel.SessionModel(
                client_id=config.CLIENT_ID, secret_key=config.SECRET_KEY,
                redirect_uri=config.REDIRECT_URI, response_type='code', grant_type='authorization_code'
            )
            auth_url = session.generate_authcode()
            self.logger.log_console("AUTH", f"1. Go to this URL and log in: {auth_url}")
            self.logger.log_console("AUTH", "2. After logging in, copy the 'auth_code' from the redirected URL.")
            auth_code = input("3. Enter the auth_code here: ")
            session.set_token(auth_code)
            response = session.generate_token()
            
            if response.get("access_token"):
                access_token = response["access_token"]
                with open(config.TOKEN_FILE, 'w') as f: f.write(access_token)
                self.logger.log_console("SUCCESS", "New access token generated and saved.")
                return access_token
            return None
        except Exception as e:
            self.logger.log_console("FATAL", f"An error occurred during authentication: {e}")
            return None

    def connect_and_load_history(self):
        if not self.access_token:
            self.access_token = self._generate_new_token()
            if not self.access_token: return False

        try:
            self.fyers_rest_client = fyersModel.FyersModel(
                client_id=config.CLIENT_ID, is_async=False, token=self.access_token, log_path=os.getcwd()
            )
            if self.fyers_rest_client.get_profile()['s'] != 'ok': raise Exception("Token validation failed.")
            self.logger.log_console("SUCCESS", "Existing access token is valid.")
        except Exception:
            self.logger.log_console("WARN", "Token invalid/expired. Re-authenticating.")
            if os.path.exists(config.TOKEN_FILE): os.remove(config.TOKEN_FILE)
            self.access_token = self._generate_new_token()
            if not self.access_token: return False
            self.fyers_rest_client = fyersModel.FyersModel(
                client_id=config.CLIENT_ID, is_async=False, token=self.access_token, log_path=os.getcwd()
            )
        
        self.logger.log_console("INFO", "Loading historical 15-min data for all symbols...")
        for symbol in self.symbols:
            hist_df = self.get_historical_data(symbol, resolution='15', days_of_data=15)
            if not hist_df.empty:
                self.candles_15min[symbol] = hist_df.set_index('datetime')
        self.logger.log_console("SUCCESS", "Historical 15-min data loaded.")

        try:
            self.fyers_ws = data_ws.FyersDataSocket(
                access_token=f"{config.CLIENT_ID}:{self.access_token}",
                log_path=os.getcwd(), on_connect=self._on_connect,
                on_error=self._on_error, on_close=self._on_close, on_message=self._on_message
            )
            ws_thread = threading.Thread(target=self.fyers_ws.connect, daemon=True)
            ws_thread.start()
            
            self.logger.log_console("INFO", "Waiting for WebSocket connection to initialize...")
            wait_cycles = 0
            while not self.is_connected and wait_cycles < 15: # Increased timeout
                time.sleep(1)
                wait_cycles += 1
            
            if not self.is_connected:
                self.logger.log_console("FATAL", "WebSocket connection did not initialize in time.")
                return False
            
            subscription_types = ["symbolData", "SymbolUpdate", "depthData", "DepthUpdate"]
            self.logger.log_console("INFO", "Connection confirmed. Sending subscription requests...")
            for stype in subscription_types:
                try:
                    self.fyers_ws.subscribe(symbols=self.symbols, data_type=stype)
                except Exception:
                    self.logger.log_console("WARN", f"Subscription for data_type='{stype}' might not be supported but proceeding.")
            
            return True

        except Exception as e:
            self.logger.log_console("ERROR", f"Failed to connect to Fyers WebSocket: {e}")
            return False

    def stop(self):
        self.logger.log_console("INFO", "Stopping WebSocket connection...")
        if self.fyers_ws:
            self.fyers_ws.unsubscribe(symbols=self.symbols)
            self.fyers_ws.close_connection()

    def get_historical_data(self, symbol, resolution='D', days_of_data=100):
        if self.fyers_rest_client is None: return pd.DataFrame()
        to_date = datetime.now(self.timezone)
        from_date = to_date - timedelta(days=days_of_data)
        data = { "symbol": symbol, "resolution": resolution, "date_format": "1",
            "range_from": from_date.strftime('%Y-%m-%d'), "range_to": to_date.strftime('%Y-%m-%d'), "cont_flag": "1" }
        try:
            response = self.fyers_rest_client.history(data=data)
            if response.get("s") == "ok":
                df = pd.DataFrame(response.get('candles', []), columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s').dt.tz_localize('UTC').dt.tz_convert(self.timezone)
                return df
            return pd.DataFrame()
        except Exception as e:
            self.logger.log_console("ERROR", f"Exception in get_historical_data for {symbol}: {e}")
            return pd.DataFrame()

    def _aggregate_tick(self, ts, symbol, price, volume):
        current_minute = ts.replace(second=0, microsecond=0)
        self.candles_1min_data.append({
            'datetime': current_minute, 'symbol': symbol, 'open': price, 'high': price, 
            'low': price, 'close': price, 'volume': volume
        })
        if current_minute.minute % 15 == 0:
            if self.last_15min_resample_time != current_minute:
                self.last_15min_resample_time = current_minute
                self._resample_to_15min(current_minute)

    def _resample_to_15min(self, current_boundary_time):
        if not self.candles_1min_data: return
        df_1min = pd.DataFrame(self.candles_1min_data)
        df_1min['datetime'] = pd.to_datetime(df_1min['datetime'])
        df_1min = df_1min.set_index('datetime')
        
        end_time = current_boundary_time
        start_time = end_time - timedelta(minutes=15)
        
        interval_df = df_1min[(df_1min.index >= start_time) & (df_1min.index < end_time)]
        if interval_df.empty: return

        agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        resampled = interval_df.groupby('symbol').resample('15T', label='left', closed='left').agg(agg_rules)
        
        for symbol, group in resampled.groupby(level=0):
            if group.empty: continue
            
            candle_timestamp = group.index[-1][1]
            candle_15min_data = group.iloc[-1].to_dict()
            
            new_candle_df = pd.DataFrame([candle_15min_data], index=[candle_timestamp])
            new_candle_df.index.name = 'datetime'

            hist_df = self.candles_15min[symbol]
            if not hist_df.empty and candle_timestamp in hist_df.index:
                hist_df = hist_df.drop(candle_timestamp)

            self.candles_15min[symbol] = pd.concat([hist_df, new_candle_df])
            
            self.event_bus.publish('CANDLE_CLOSED_15MIN', {'symbol': symbol})
            self.logger.log_console("DEBUG", f"15m candle for {symbol} closed @ {start_time.strftime('%H:%M')}")
        
        self.candles_1min_data = [d for d in self.candles_1min_data if d['datetime'] >= current_boundary_time]

