# ------------------------------------------------------------------------------------------------
# market_data_test.py - A Zero-Trust Diagnostic Tool for Live Markets
# ------------------------------------------------------------------------------------------------
#
# ARCHITECTURAL UPDATE: This script now incorporates a "brute-force" subscription
# model and intelligent message parsing, based on the proven working example.
# It attempts to subscribe to all relevant data types and only prints clean,
# formatted tick data.
#
# ------------------------------------------------------------------------------------------------

import os
import time
import json
import threading
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

# --- Import config ---
try:
    import config
except ImportError:
    print("FATAL ERROR: config.py not found. Please ensure it's in the same directory.")
    exit()

# --- CONFIGURATION FOR THIS TEST ---
# Using high-volume commodity futures to ensure there is always live data.
# Note: Using the mini contract as it may have higher tick frequency.
TEST_SYMBOLS = [
    "MCX:CRUDEOILM25SEPFUT",
    "MCX:GOLDM25OCTFUT"
]

# --- Global state variables ---
is_connected = False
fyers_ws = None

# --- WebSocket Callbacks ---

def on_connect():
    """Callback executed upon a successful WebSocket connection."""
    global is_connected
    print(f"[{time.strftime('%H:%M:%S')}] SUCCESS: Fyers WebSocket connected.")
    is_connected = True

def on_error(msg):
    """Callback to handle WebSocket errors."""
    print(f"[{time.strftime('%H:%M:%S')}] ERROR: WebSocket Error: {msg}")

def on_close(msg=""):
    """Callback for when the WebSocket connection is closed."""
    global is_connected
    print(f"[{time.strftime('%H:%M:%S')}] INFO: WebSocket closed. Reason: {msg}")
    is_connected = False

def on_message(msg):
    """Process incoming WebSocket messages and print only clean tick info."""
    try:
        # The Fyers library may return a dict or a JSON string
        data = json.loads(msg) if isinstance(msg, str) else msg
        if not isinstance(data, dict):
            # Ignore non-dictionary messages
            return

        ts = time.strftime('%H:%M:%S')

        # 'sf' = symbol feed (the most common type for ticks)
        if data.get("type") == "sf":
            symbol = data.get("symbol")
            ltp = data.get("ltp")
            qty = data.get("last_traded_qty")
            print(f"[{ts}] {symbol} | LTP={ltp} | QTY={qty}")

        # 'dp' = depth packet (can also contain last traded price)
        elif data.get("type") == "dp":
            symbol = data.get("symbol")
            ltp = data.get("ltp")
            qty = data.get("last_traded_qty")
            # Only print if the depth packet contains a valid trade update
            if ltp is not None and qty is not None:
                print(f"[{ts}] {symbol} | LTP={ltp} | QTY={qty}")
                
    except Exception as e:
        # This helps debug if the message format is unexpected
        print(f"[{time.strftime('%H:%M:%S')}] PARSE ERROR: {msg} | {e}")


# --- Authentication Logic ---

def read_token_from_file():
    if os.path.exists(config.TOKEN_FILE):
        with open(config.TOKEN_FILE, 'r') as f: return f.read().strip()
    return None

def generate_new_token():
    try:
        session = fyersModel.SessionModel(
            client_id=config.CLIENT_ID, secret_key=config.SECRET_KEY,
            redirect_uri=config.REDIRECT_URI, response_type='code', grant_type='authorization_code'
        )
        auth_url = session.generate_authcode()
        print(f"[{time.strftime('%H:%M:%S')}] AUTH: 1. Go to this URL and log in: {auth_url}")
        print(f"[{time.strftime('%H:%M:%S')}] AUTH: 2. After logging in, copy the 'auth_code' from the redirected URL.")
        auth_code = input("3. Enter the auth_code here: ")
        session.set_token(auth_code)
        response = session.generate_token()
        
        if response.get("access_token"):
            token = response["access_token"]
            with open(config.TOKEN_FILE, 'w') as f: f.write(token)
            print(f"[{time.strftime('%H:%M:%S')}] SUCCESS: New access token generated and saved.")
            return token
        return None
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] FATAL: An error occurred during authentication: {e}")
        return None

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Fyers Market Data Connection Test (Zero-Trust) ---")
    
    access_token = read_token_from_file() or generate_new_token()
    if not access_token:
        print(f"[{time.strftime('%H:%M:%S')}] FATAL: Could not get access token. Exiting.")
        exit()
    
    print(f"[{time.strftime('%H:%M:%S')}] INFO: Access token loaded successfully.")

    fyers_ws = data_ws.FyersDataSocket(
        access_token=f"{config.CLIENT_ID}:{access_token}",
        log_path=os.getcwd(), on_connect=on_connect,
        on_error=on_error, on_close=on_close, on_message=on_message
    )
    
    ws_thread = threading.Thread(target=fyers_ws.connect, daemon=True)
    ws_thread.start()
    
    print(f"[{time.strftime('%H:%M:%S')}] INFO: Waiting for connection confirmation...")
    wait_cycles = 0
    while not is_connected and wait_cycles < 10:
        time.sleep(1)
        wait_cycles += 1
        
    if not is_connected:
        print(f"[{time.strftime('%H:%M:%S')}] FATAL: WebSocket did not connect in time. Exiting.")
        exit()
    
    # --- Brute-force subscribe to all possible data types ---
    # This ensures we receive data regardless of the specific stream the API uses
    subscription_types = ["symbolData", "SymbolUpdate", "depthData", "DepthUpdate"]
    print(f"[{time.strftime('%H:%M:%S')}] INFO: Connection confirmed. Sending subscription requests...")
    for stype in subscription_types:
        try:
            fyers_ws.subscribe(symbols=TEST_SYMBOLS, data_type=stype)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] WARN: Subscription failed for data_type='{stype}': {e}")
    
    print(f"[{time.strftime('%H:%M:%S')}] INFO: Subscription requests sent. Now monitoring for live data...")
    print("(Press Ctrl+C to exit)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] INFO: Ctrl+C detected. Shutting down...")
        if fyers_ws and is_connected:
            fyers_ws.unsubscribe(symbols=TEST_SYMBOLS)
            fyers_ws.close_connection()
        print(f"[{time.strftime('%H:%M:%S')}] INFO: Test complete.")

