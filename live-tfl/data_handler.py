# ------------------------------------------------------------------------------------------------
# data_handler.py - Connects to Fyers, Fetches Ticks, and Builds Candles
# ------------------------------------------------------------------------------------------------
#
# This component is responsible for all market data interactions.
# - Handles authentication with Fyers.
# - Connects to the websocket for live tick data.
# - Manages the subscription to the required symbols.
# - Aggregates incoming ticks into 1-minute and 15-minute candles.
# - Publishes 'TICK' and 'CANDLE_CLOSED_15MIN' events to the Event Bus.
# - Fetches historical data needed for indicator calculations.
#
# ------------------------------------------------------------------------------------------------

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
# --- FIX: Using the correct import path for the FyersDataSocket class ---
from fyers_apiv3.FyersWebsocket.data_ws import FyersDataSocket
import pandas_ta as ta

import config

class DataHandler:
    def __init__(self, symbols, event_bus, logger):
        self.symbols = symbols
        self.event_bus = event_bus
        self.logger = logger
        
        # Internal state for candle aggregation
        self.candles_1min = {symbol: [] for symbol in self.symbols}
        self.candles_15min = {symbol: pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume']) for symbol in self.symbols}
        self.last_tick_prices = {symbol: None for symbol in self.symbols}

        self.fyers_rest_client = None
        self.fyers_ws = None
        self.access_token = self._read_token_from_file() # Read token initially

    def _read_token_from_file(self):
        """Reads the access token from the file if it exists."""
        if os.path.exists(config.TOKEN_FILE):
            with open(config.TOKEN_FILE, 'r') as f:
                return f.read().strip()
        return None

    def _generate_new_token(self):
        """Handles the interactive Fyers authentication flow to get a new access token."""
        try:
            self.logger.log_console("AUTH", "Starting authentication flow to get a new access token...")
            session = fyersModel.SessionModel(
                client_id=config.CLIENT_ID,
                secret_key=config.SECRET_KEY,
                redirect_uri=config.REDIRECT_URI,
                response_type='code',
                grant_type='authorization_code'
            )
            auth_url = session.generate_authcode()
            self.logger.log_console("AUTH", f"1. Go to this URL and log in: {auth_url}")
            self.logger.log_console("AUTH", "2. After logging in, copy the 'auth_code' from the redirected URL.")
            auth_code = input("3. Enter the auth_code here: ")
            session.set_token(auth_code)
            response = session.generate_token()
            
            if response.get("access_token"):
                access_token = response["access_token"]
                with open(config.TOKEN_FILE, 'w') as f:
                    f.write(access_token)
                self.logger.log_console("SUCCESS", "New access token generated and saved.")
                return access_token
            else:
                self.logger.log_console("FATAL", f"Failed to generate access token: {response}")
                return None
        except Exception as e:
            self.logger.log_console("FATAL", f"An error occurred during authentication: {e}")
            return None

    def connect(self):
        """Initializes clients and handles token expiration and renewal."""
        if not self.access_token:
            self.logger.log_console("INFO", "Access token not found.")
            self.access_token = self._generate_new_token()
            if not self.access_token: return False

        # --- ENHANCED LOGIC: Attempt connection and re-authenticate on failure ---
        try:
            self.logger.log_console("INFO", "Attempting to validate existing access token...")
            self.fyers_rest_client = fyersModel.FyersModel(
                client_id=config.CLIENT_ID, is_async=False, token=self.access_token, log_path=os.getcwd()
            )
            # A simple API call to check if the token is valid
            if self.fyers_rest_client.get_profile()['s'] != 'ok':
                raise Exception("Token validation failed.")
            self.logger.log_console("SUCCESS", "Existing access token is valid.")

        except Exception:
            self.logger.log_console("WARN", "Existing access token is invalid or expired. Deleting old token and re-authenticating.")
            if os.path.exists(config.TOKEN_FILE):
                os.remove(config.TOKEN_FILE)
            
            self.access_token = self._generate_new_token()
            if not self.access_token: return False
            
            # Retry initializing the client with the new token
            self.fyers_rest_client = fyersModel.FyersModel(
                client_id=config.CLIENT_ID, is_async=False, token=self.access_token, log_path=os.getcwd()
            )

        # --- WebSocket client setup (proceeds only after successful REST client init) ---
        try:
            self.logger.log_console("INFO", "Attempting to connect to Fyers WebSocket...")
            self.fyers_ws = FyersDataSocket(
                access_token=f"{config.CLIENT_ID}:{self.access_token}",
                log_path=os.getcwd()
            )
            self.fyers_ws.on_connect = lambda: self.logger.log_console("SUCCESS", "Fyers WebSocket connected.")
            self.fyers_ws.on_error = lambda msg: self.logger.log_console("ERROR", f"WebSocket Error: {msg}")
            self.fyers_ws.on_close = lambda: self.logger.log_console("INFO", "WebSocket connection closed.")
            self.fyers_ws.on_message = self.on_message
            self.fyers_ws.connect()
            
            self.fyers_ws.subscribe(symbols=self.symbols, data_type="symbolData")
            return True

        except Exception as e:
            self.logger.log_console("ERROR", f"Failed to connect to Fyers WebSocket: {e}")
            return False

    def get_historical_data(self, symbol, resolution='D', days_of_data=100):
        """Fetches historical OHLC data with error handling."""
        if self.fyers_rest_client is None:
            self.logger.log_console("ERROR", f"REST client not initialized. Cannot fetch historical data for {symbol}.")
            return pd.DataFrame()

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_of_data)
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": from_date.strftime('%Y-%m-%d'),
            "range_to": to_date.strftime('%Y-%m-%d'),
            "cont_flag": "1"
        }
        try:
            response = self.fyers_rest_client.history(data=data)
            if response.get("s") == "ok":
                candles = response.get('candles', [])
                if not candles: return pd.DataFrame()
                df = pd.DataFrame(candles, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                return df
            else:
                self.logger.log_console("ERROR", f"API error fetching history for {symbol}: {response.get('message', 'Unknown error')}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.log_console("ERROR", f"Exception in get_historical_data for {symbol}: {e}")
            return pd.DataFrame()
            
    def on_message(self, msg):
        """Callback to process incoming tick data from the websocket."""
        try:
            # Fyers sends a list of ticks in a single message
            ticks = msg.get('d', {}).get('7208', [])
            for tick in ticks:
                symbol = tick['v']['symbol']
                ltp = tick['v']['ltp']
                
                if symbol in self.last_tick_prices:
                    self.last_tick_prices[symbol] = ltp
                    self.event_bus.publish('TICK', {'symbol': symbol, 'price': ltp})
                    self._aggregate_to_1min(symbol, ltp, tick.get('v', {}).get('volume', 0))
        except Exception as e:
            self.logger.log_console("ERROR", f"Error processing tick message: {msg} | Exception: {e}")


    def _aggregate_to_1min(self, symbol, price, volume):
        """Aggregates ticks into 1-minute candles."""
        now = datetime.now()
        current_minute = now.replace(second=0, microsecond=0)
        
        candle_list = self.candles_1min[symbol]
        
        if not candle_list or candle_list[-1]['datetime'] != current_minute:
            # New 1-min candle starts
            new_candle = {
                'datetime': current_minute,
                'open': price, 'high': price,
                'low': price, 'close': price, 'volume': int(volume)
            }
            candle_list.append(new_candle)
            
            # Check if a 15-min candle just closed
            if len(candle_list) > 1 and current_minute.minute > 0 and current_minute.minute % 15 == 0:
                 self._aggregate_to_15min(symbol, candle_list[-2]['datetime'])
        else:
            # Update current 1-min candle
            candle_list[-1]['high'] = max(candle_list[-1]['high'], price)
            candle_list[-1]['low'] = min(candle_list[-1]['low'], price)
            candle_list[-1]['close'] = price
            candle_list[-1]['volume'] += int(volume)

    def _aggregate_to_15min(self, symbol, last_minute):
        """Aggregates the last 15 1-minute candles into a single 15-minute candle."""
        if len(self.candles_1min[symbol]) < 15:
             return # Not enough data yet

        # Get the last 15 candles for aggregation
        relevant_candles = self.candles_1min[symbol][-15:]
        fifteen_min_candles_df = pd.DataFrame(relevant_candles)
        
        if fifteen_min_candles_df.empty:
            return

        candle_15min_time = last_minute.replace(minute=(last_minute.minute // 15) * 15)
        
        agg_candle = {
            'datetime': candle_15min_time,
            'open': fifteen_min_candles_df['open'].iloc[0],
            'high': fifteen_min_candles_df['high'].max(),
            'low': fifteen_min_candles_df['low'].min(),
            'close': fifteen_min_candles_df['close'].iloc[-1],
            'volume': fifteen_min_candles_df['volume'].sum()
        }
        
        new_row = pd.DataFrame([agg_candle])
        self.candles_15min[symbol] = pd.concat([self.candles_15min[symbol], new_row], ignore_index=True)
        
        self.event_bus.publish('CANDLE_CLOSED_15MIN', {'symbol': symbol, 'candle': agg_candle})
        self.logger.log_console("DEBUG", f"15m candle closed for {symbol} @ {candle_15min_time.strftime('%H:%M')}")

