# main_simulator.py
# This single-file simulator contains both the configuration and the backtesting engine.
# It is designed to be run from within the 'tfl' directory.
# ENHANCEMENT: Replaced Market Regime filter with a directional MVWAP filter.

import pandas as pd
import os
import datetime
import logging
import sys
from pytz import timezone

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# -- PROJECT ROOT DIRECTORY --
# Assumes this script is inside a 'tfl' subfolder.
# Navigates one level up to get to the project's root directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# -- DATE RANGE FOR BACKTEST --
# Defines the specific period to run the simulation on.
# Format: 'YYYY-MM-DD'
START_DATE = '2018-01-01'
END_DATE = '2025-01-31'

# -- STRATEGY CONFIGURATION --
# The name of the strategy, used for organizing log files.
STRATEGY_NAME = "TrafficLight-Manny"
# The risk-to-reward ratio for setting the take-profit target.
RISK_REWARD_RATIO = 1.5
# The number of past candles to look at to identify the trade setup pattern.
PATTERN_LOOKBACK_CANDLES = 5

# -- SIMULATOR MODE --
# "INTRADAY": All open positions are automatically closed at the EOD_TIME.
# "POSITIONAL": Positions can be held overnight across multiple days.
TRADING_MODE = "POSITIONAL"

# -- MVWAP FILTER --
# If enabled, trades will only be taken in the direction of the MVWAP trend.
# - Long trades are only allowed if the closing price is ABOVE the MVWAP.
# - Short trades are only allowed if the closing price is BELOW the MVWAP.
USE_MVWAP_FILTER = True
# The period for the Moving Volume Weighted Average Price (MVWAP) calculation.
# This must match a column calculated by the data pipeline (e.g., 'mvwap_50').
MVWAP_PERIOD = 50

# -- DATA & FILE PATHS --
# Path to the 15-minute data with pre-calculated indicators.
DATA_DIRECTORY = os.path.join(ROOT_DIR, "data", "universal_processed", "15min")
# Path to the CSV file containing the list of symbols to backtest.
SYMBOL_LIST_PATH = os.path.join(ROOT_DIR, "nifty200_fno.csv")

# -- ADDITIONAL SYMBOLS --
# A list of extra symbols to add to the backtest, in addition to the list from the CSV.
# Useful for adding indices like NIFTY50 and BANKNIFTY.
ADDITIONAL_SYMBOLS = ['NIFTY50-INDEX', 'NIFTYBANK-INDEX']

# -- LOGGING CONFIGURATION --
# The main folder where all backtest logs will be stored.
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")

# -- TRADING SESSION --
# The time to close all open positions if TRADING_MODE is "INTRADAY".
EOD_TIME = "15:15"

# ==============================================================================
# --- BACKTESTING ENGINE ---
# ==============================================================================

def setup_logging(log_dir):
    """Configures the logging for the simulation summary."""
    summary_file_path = os.path.join(log_dir, 'summary.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(summary_file_path),
            logging.StreamHandler()
        ]
    )

def log_configs():
    """Logs the configuration settings used for the backtest run."""
    logging.info("--- Starting Backtest Simulation ---")
    logging.info("Configuration Settings:")
    logging.info(f"  - Strategy Name: {STRATEGY_NAME}")
    logging.info(f"  - Backtest Period: {START_DATE} to {END_DATE}")
    logging.info(f"  - Risk/Reward Ratio: 1:{RISK_REWARD_RATIO}")
    logging.info(f"  - Pattern Lookback Period: {PATTERN_LOOKBACK_CANDLES} candles")
    logging.info(f"  - Trading Mode: {TRADING_MODE}")
    logging.info(f"  - Use MVWAP Filter: {USE_MVWAP_FILTER} (Period: {MVWAP_PERIOD})")
    logging.info("-" * 30)

def find_trade_setup(df_slice):
    """
    Analyzes a slice of the dataframe (T-N to T-1) to find a valid trade setup.
    """
    last_candle = df_slice.iloc[-1]
    if last_candle['close'] > last_candle['open']:
        red_candles_slice = df_slice.iloc[:-1]
        if not red_candles_slice.empty and all(red_candles_slice['close'] < red_candles_slice['open']):
            return 'LONG', df_slice['high'].max(), df_slice['low'].min()
    if last_candle['close'] < last_candle['open']:
        green_candles_slice = df_slice.iloc[:-1]
        if not green_candles_slice.empty and all(green_candles_slice['close'] > green_candles_slice['open']):
            return 'SHORT', df_slice['high'].max(), df_slice['low'].min()
    return None, None, None

def run_backtest(df, symbol):
    """
    Main backtesting function that iterates through the data candle by candle.
    """
    trades = []
    active_trade = None

    for i in range(PATTERN_LOOKBACK_CANDLES, len(df)):
        current_candle = df.iloc[i]
        
        if active_trade:
            exit_reason, exit_price = None, None
            if active_trade['direction'] == 'LONG':
                if current_candle['low'] <= active_trade['sl']: exit_reason, exit_price = 'SL_HIT', active_trade['sl']
                elif current_candle['high'] >= active_trade['tp']: exit_reason, exit_price = 'TP_HIT', active_trade['tp']
            elif active_trade['direction'] == 'SHORT':
                if current_candle['high'] >= active_trade['sl']: exit_reason, exit_price = 'SL_HIT', active_trade['sl']
                elif current_candle['low'] <= active_trade['tp']: exit_reason, exit_price = 'TP_HIT', active_trade['tp']
            if TRADING_MODE == 'INTRADAY' and current_candle.name.strftime('%H:%M') == EOD_TIME and not exit_reason:
                exit_reason, exit_price = 'EOD_EXIT', current_candle['close']
            if exit_reason:
                active_trade.update({'exit_time': current_candle.name, 'exit_price': exit_price, 'exit_reason': exit_reason, 'pnl': (exit_price - active_trade['entry_price']) if active_trade['direction'] == 'LONG' else (active_trade['entry_price'] - exit_price)})
                trades.append(active_trade)
                active_trade = None

        if not active_trade:
            df_slice = df.iloc[i - PATTERN_LOOKBACK_CANDLES:i]
            direction, pattern_high, pattern_low = find_trade_setup(df_slice)

            if direction:
                # --- NEW MVWAP FILTER ---
                if USE_MVWAP_FILTER:
                    mvwap_col = f'mvwap_{MVWAP_PERIOD}'
                    # Ensure the MVWAP data exists for the current candle
                    if mvwap_col not in df.columns or pd.isna(current_candle[mvwap_col]):
                        continue 
                    
                    price_above_mvwap = current_candle['close'] > current_candle[mvwap_col]
                    
                    # If price is above MVWAP, only allow LONG trades.
                    if price_above_mvwap and direction != 'LONG':
                        continue
                    # If price is below MVWAP, only allow SHORT trades.
                    if not price_above_mvwap and direction != 'SHORT':
                        continue

                if direction == 'LONG' and current_candle['high'] > pattern_high:
                    entry_price, sl = pattern_high, pattern_low
                    tp = entry_price + (entry_price - sl) * RISK_REWARD_RATIO
                    active_trade = {'symbol': symbol, 'direction': 'LONG', 'entry_time': current_candle.name, 'entry_price': entry_price, 'sl': sl, 'tp': tp}
                    if current_candle['low'] <= sl:
                        active_trade.update({'exit_time': current_candle.name, 'exit_price': sl, 'exit_reason': 'SL_HIT_ON_ENTRY', 'pnl': sl - entry_price})
                        trades.append(active_trade)
                        active_trade = None
                elif direction == 'SHORT' and current_candle['low'] < pattern_low:
                    entry_price, sl = pattern_low, pattern_high
                    tp = entry_price - (sl - entry_price) * RISK_REWARD_RATIO
                    active_trade = {'symbol': symbol, 'direction': 'SHORT', 'entry_time': current_candle.name, 'entry_price': entry_price, 'sl': sl, 'tp': tp}
                    if current_candle['high'] >= sl:
                        active_trade.update({'exit_time': current_candle.name, 'exit_price': sl, 'exit_reason': 'SL_HIT_ON_ENTRY', 'pnl': entry_price - sl})
                        trades.append(active_trade)
                        active_trade = None
    return trades

def calculate_and_log_metrics(all_trades_df, scope_name):
    """Calculates and logs performance metrics."""
    if all_trades_df.empty:
        logging.info(f"No trades executed for {scope_name}.")
        return
    logging.info(f"\n--- Performance Metrics for {scope_name} ---")
    total_trades = len(all_trades_df)
    wins = all_trades_df[all_trades_df['pnl'] > 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = wins['pnl'].sum()
    gross_loss = all_trades_df[all_trades_df['pnl'] <= 0]['pnl'].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
    logging.info(f"Total Trades: {total_trades}, Win Rate: {win_rate:.2f}%, Profit Factor: {profit_factor:.2f}, Net PnL: {all_trades_df['pnl'].sum():.2f}")
    for direction in ['LONG', 'SHORT']:
        dir_trades = all_trades_df[all_trades_df['direction'] == direction]
        if not dir_trades.empty:
            dir_wins = dir_trades[dir_trades['pnl'] > 0]
            dir_win_rate = (len(dir_wins) / len(dir_trades)) * 100
            logging.info(f"  - {direction} Trades: {len(dir_trades)}, Win Rate: {dir_win_rate:.2f}%, PnL: {dir_trades['pnl'].sum():.2f}")
    logging.info("-" * 30)

def main():
    """Main function to orchestrate the backtesting process."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(LOGS_BASE_DIR, STRATEGY_NAME, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)
    log_configs()

    try:
        symbols_df = pd.read_csv(SYMBOL_LIST_PATH)
        symbols_from_csv = symbols_df['symbol'].tolist()
    except FileNotFoundError:
        logging.warning(f"Symbol list file not found at: {SYMBOL_LIST_PATH}. Proceeding with additional symbols only.")
        symbols_from_csv = []

    # Combine the list from the CSV with the additional symbols, removing duplicates.
    symbols_to_process = sorted(list(set(symbols_from_csv + ADDITIONAL_SYMBOLS)))
        
    logging.info(f"Loaded {len(symbols_from_csv)} symbols from {SYMBOL_LIST_PATH}")
    logging.info(f"Added {len(ADDITIONAL_SYMBOLS)} additional symbols.")
    logging.info(f"Total unique symbols to process: {len(symbols_to_process)}")
        
    all_symbols_trades = []
    
    try:
        tz = timezone('Asia/Kolkata')
        start_dt = tz.localize(datetime.datetime.strptime(START_DATE, '%Y-%m-%d'))
        end_dt = tz.localize(datetime.datetime.strptime(END_DATE, '%Y-%m-%d')) + datetime.timedelta(days=1)
    except ValueError as e:
        logging.error(f"Date parsing error: {e}. Check START_DATE/END_DATE format ('YYYY-MM-DD')."); sys.exit(1)

    for symbol in symbols_to_process:
        data_path = os.path.join(DATA_DIRECTORY, f"{symbol}_15min_with_indicators.csv")
        if not os.path.exists(data_path):
            logging.warning(f"Data file not found for {symbol}. Skipping."); continue

        try:
            df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
            df.index = df.index.tz_localize(tz)
            df_filtered = df[(df.index >= start_dt) & (df.index < end_dt)].copy()
            if df_filtered.empty: continue

            logging.info(f"\nProcessing symbol: {symbol}...")
            trades = run_backtest(df_filtered, symbol)
            if trades:
                trades_df = pd.DataFrame(trades)
                calculate_and_log_metrics(trades_df, symbol)
                all_symbols_trades.append(trades_df)
        except Exception as e:
            logging.error(f"An error occurred while processing {symbol}: {e}")

    if all_symbols_trades:
        final_trades_df = pd.concat(all_symbols_trades, ignore_index=True)
        trade_log_path = os.path.join(log_dir, 'trade_log.csv')
        final_trades_df.to_csv(trade_log_path, index=False)
        logging.info(f"\n--- Overall Backtest Summary ---")
        calculate_and_log_metrics(final_trades_df, "ALL_SYMBOLS_COMBINED")
        logging.info(f"Full trade log saved to: {trade_log_path}")
    else:
        logging.info("No trades were executed across any symbols in the specified date range.")
    logging.info("--- Backtest Simulation Finished ---")

if __name__ == "__main__":
    main()
