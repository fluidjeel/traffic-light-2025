# shorts_simulator.py
#
# Description:
# A dedicated backtester for analyzing and optimizing the short side of the strategy.
# ENHANCEMENT: Added an optional MVWAP filter as an additional confirmation layer.

import pandas as pd
import os
import datetime
import logging
import sys
import numpy as np
from pytz import timezone
from multiprocessing import Pool, cpu_count
from functools import partial

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# -- PROJECT ROOT DIRECTORY --
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# -- DATE RANGE FOR BACKTEST --
START_DATE = '2018-01-01'
END_DATE = '2025-01-31'

# -- STRATEGY CONFIGURATION --
STRATEGY_NAME = "TrafficLight-Manny-SHORTS_ONLY"
RISK_REWARD_RATIO = 10.0

# --- DYNAMIC PATTERN CONFIGURATION ---
MAX_PATTERN_LOOKBACK = 10
MIN_CONSECUTIVE_CANDLES = 1


# -- SIMULATOR MODE --
TRADING_MODE = "POSITIONAL"

# -- TRADE MANAGEMENT --
USE_ATR_TRAILING_STOP = True
ATR_TS_PERIOD = 14
ATR_TS_MULTIPLIER = 2.0

# -- BREAKEVEN STOP LOGIC --
USE_BREAKEVEN_STOP = True
BREAKEVEN_TRIGGER_R = 0.5
BREAKEVEN_PROFIT_R = 0.1

# -- MULTI-STAGE TRAILING STOP --
USE_MULTI_STAGE_TS = True
AGGRESSIVE_TS_TRIGGER_R = 3.0
AGGRESSIVE_TS_MULTIPLIER = 1.0

# --- DYNAMIC UNIVERSE FILTER ---
USE_DYNAMIC_UNIVERSE = False
VIX_SYMBOL = 'INDIAVIX'
VIX_THRESHOLD = 20.0
VOLATILITY_ATR_PERIOD = 14
VOLATILITY_UNIVERSE_PERCENTILE = 0.75

# --- NEW: ADDITIONAL MVWAP FILTER ---
# If enabled, this adds a final check. Only shorts stocks trading below their MVWAP.
# This is applied *after* the dynamic universe filter. It is skipped for indices.
USE_ADDITIONAL_MVWAP_FILTER = True
MVWAP_PERIOD = 50


# -- DATA & FILE PATHS --
DATA_DIRECTORY_15MIN = os.path.join(ROOT_DIR, "data", "universal_processed", "15min")
DATA_DIRECTORY_DAILY = os.path.join(ROOT_DIR, "data", "universal_processed", "daily")
SYMBOL_LIST_PATH = os.path.join(ROOT_DIR, "nifty200_fno.csv")

# -- ADDITIONAL SYMBOLS --
ADDITIONAL_SYMBOLS = ['NIFTY50-INDEX', 'NIFTYBANK-INDEX']

# -- LOGGING CONFIGURATION --
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")

# -- TRADING SESSION --
EOD_TIME = "15:15"

# ==============================================================================
# --- BACKTESTING ENGINE ---
# ==============================================================================

def setup_logging(log_dir):
    """Configures the logging for the simulation summary."""
    summary_file_path = os.path.join(log_dir, 'summary.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(summary_file_path), logging.StreamHandler()])

def log_configs():
    """Logs the configuration settings used for the backtest run."""
    logging.info("--- Starting SHORTS-ONLY Backtest Simulation ---")
    logging.info(f"  - Strategy Name: {STRATEGY_NAME}")
    logging.info(f"  - Backtest Period: {START_DATE} to {END_DATE}")
    logging.info(f"  - Max Lookback: {MAX_PATTERN_LOOKBACK}, Min Consecutive: {MIN_CONSECUTIVE_CANDLES}")
    logging.info(f"  - Using Dynamic Universe Filter: {USE_DYNAMIC_UNIVERSE} (VIX > {VIX_THRESHOLD}, Top {100 - VOLATILITY_UNIVERSE_PERCENTILE*100}% Volatility)")
    logging.info(f"  - Using Additional MVWAP Filter: {USE_ADDITIONAL_MVWAP_FILTER} (Period: {MVWAP_PERIOD})")
    logging.info("-" * 30)

def find_trade_setup(df_slice):
    """
    Scans backwards from the last candle within the lookback window to find a
    valid pattern of consecutive green candles followed by a red candle.
    """
    signal_candle = df_slice.iloc[-1]
    if signal_candle['close'] >= signal_candle['open']:
        return None, None, None

    consecutive_green_count = 0
    for i in range(len(df_slice) - 2, -1, -1):
        if df_slice.iloc[i]['close'] > df_slice.iloc[i]['open']:
            consecutive_green_count += 1
        else:
            break
            
    if consecutive_green_count >= MIN_CONSECUTIVE_CANDLES:
        pattern_candles = df_slice.iloc[-(consecutive_green_count + 1):]
        pattern_high = pattern_candles['high'].max()
        pattern_low = pattern_candles['low'].min()
        return 'SHORT', pattern_high, pattern_low
        
    return None, None, None

def run_backtest(df, symbol, daily_filters, daily_data_map):
    """Main backtesting function that iterates through the data candle by candle."""
    trades = []
    active_trade = None
    atr_col = f'atr_{ATR_TS_PERIOD}'
    is_index = "-INDEX" in symbol
    
    symbol_daily_data = daily_data_map.get(symbol)

    for i in range(MAX_PATTERN_LOOKBACK, len(df)):
        current_candle = df.iloc[i]
        current_date = current_candle.name.normalize()

        if active_trade:
            # --- MFE and MAE Tracking ---
            active_trade['mfe'] = max(active_trade['mfe'], active_trade['entry_price'] - current_candle['low'])
            active_trade['mae'] = max(active_trade['mae'], current_candle['high'] - active_trade['entry_price'])

            # --- BREAKEVEN STOP LOGIC ---
            if USE_BREAKEVEN_STOP and not active_trade['be_activated']:
                if current_candle['low'] <= active_trade['be_trigger_price']:
                    active_trade['sl'] = active_trade['be_target_price']
                    active_trade['be_activated'] = True
            
            # --- MULTI-STAGE TRAILING STOP ACTIVATION ---
            if USE_MULTI_STAGE_TS:
                if active_trade['initial_risk'] > 0:
                    current_profit_r = (active_trade['entry_price'] - current_candle['low']) / active_trade['initial_risk']
                    if current_profit_r >= AGGRESSIVE_TS_TRIGGER_R:
                        active_trade['current_ts_multiplier'] = AGGRESSIVE_TS_MULTIPLIER

            # --- ATR Trailing Stop Logic ---
            if USE_ATR_TRAILING_STOP and atr_col in df.columns and pd.notna(current_candle[atr_col]):
                current_atr = current_candle[atr_col]
                ts_multiplier = active_trade['current_ts_multiplier']
                new_trailing_stop = current_candle['low'] + (current_atr * ts_multiplier)
                active_trade['sl'] = min(active_trade['sl'], new_trailing_stop)

            # --- Exit Logic ---
            exit_reason, exit_price = None, None
            if current_candle['high'] >= active_trade['sl']: exit_reason, exit_price = 'SL_HIT', active_trade['sl']
            elif current_candle['low'] <= active_trade['tp']: exit_reason, exit_price = 'TP_HIT', active_trade['tp']
            
            if TRADING_MODE == 'INTRADAY' and current_candle.name.strftime('%H:%M') == EOD_TIME and not exit_reason:
                exit_reason, exit_price = 'EOD_EXIT', current_candle['close']
            
            if exit_reason:
                active_trade.update({'exit_time': current_candle.name, 'exit_price': exit_price, 'exit_reason': exit_reason, 'pnl': (active_trade['entry_price'] - exit_price)})
                trades.append(active_trade)
                active_trade = None

        if not active_trade:
            # --- DYNAMIC UNIVERSE FILTER CHECK ---
            if USE_DYNAMIC_UNIVERSE:
                daily_info = daily_filters.get(current_date)
                if not daily_info or not daily_info['vix_ok'] or symbol not in daily_info['hotlist']:
                    continue

            df_slice = df.iloc[i - MAX_PATTERN_LOOKBACK:i]
            direction, pattern_high, pattern_low = find_trade_setup(df_slice)

            if direction == 'SHORT':
                # --- ADDITIONAL MVWAP FILTER for Stocks ---
                if USE_ADDITIONAL_MVWAP_FILTER and not is_index:
                    mvwap_col = f'mvwap_{MVWAP_PERIOD}'
                    if mvwap_col not in df.columns or pd.isna(current_candle[mvwap_col]):
                        continue
                    if current_candle['close'] > current_candle[mvwap_col]:
                        continue # Skip if price is above MVWAP

                if current_candle['low'] < pattern_low:
                    entry_price, sl = pattern_low, pattern_high
                    initial_risk = sl - entry_price
                    if initial_risk <= 0: continue
                    
                    # --- Log Volatility and RSI at Entry ---
                    daily_volatility = np.nan
                    daily_rsi = np.nan
                    if symbol_daily_data is not None and current_date in symbol_daily_data.index:
                        daily_stats = symbol_daily_data.loc[current_date]
                        daily_volatility = daily_stats.get(f'atr_{VOLATILITY_ATR_PERIOD}_pct', np.nan)
                        daily_rsi = daily_stats.get('rsi_14', np.nan)

                    tp = entry_price - (initial_risk * RISK_REWARD_RATIO)
                    active_trade = {
                        'symbol': symbol, 'direction': 'SHORT', 'entry_time': current_candle.name, 
                        'entry_price': entry_price, 'sl': sl, 'tp': tp, 'mfe': 0, 'mae': 0,
                        'initial_risk': initial_risk, 'be_activated': False,
                        'be_trigger_price': entry_price - (initial_risk * BREAKEVEN_TRIGGER_R),
                        'be_target_price': entry_price - (initial_risk * BREAKEVEN_PROFIT_R),
                        'current_ts_multiplier': ATR_TS_MULTIPLIER,
                        'daily_vol_at_entry': daily_volatility,
                        'daily_rsi_at_entry': daily_rsi
                    }
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

def process_symbol_backtest(symbol, daily_filters, daily_data_map):
    """A wrapper function to run the backtest for a single symbol."""
    tz = timezone('Asia/Kolkata')
    start_dt = tz.localize(datetime.datetime.strptime(START_DATE, '%Y-%m-%d'))
    end_dt = tz.localize(datetime.datetime.strptime(END_DATE, '%Y-%m-%d')) + datetime.timedelta(days=1)

    data_path = os.path.join(DATA_DIRECTORY_15MIN, f"{symbol}_15min_with_indicators.parquet")
    if not os.path.exists(data_path):
        print(f"Data file not found for {symbol}. Skipping.")
        return None

    try:
        df = pd.read_parquet(data_path)
        df.index = df.index.tz_localize(tz)
        df_filtered = df[(df.index >= start_dt) & (df.index < end_dt)].copy()
        if df_filtered.empty: return None

        print(f"Processing symbol: {symbol}...")
        trades = run_backtest(df_filtered, symbol, daily_filters, daily_data_map)
        
        if trades: return pd.DataFrame(trades)
        return None
    except Exception as e:
        print(f"An error occurred while processing {symbol}: {e}")
        return None

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
        logging.warning(f"Symbol list file not found at: {SYMBOL_LIST_PATH}.")
        symbols_from_csv = []

    # Define the symbols we will actually run the backtest on
    symbols_to_trade = sorted(list(set(symbols_from_csv + ADDITIONAL_SYMBOLS)))
    
    # Define the complete set of symbols for which we need to load data.
    # This includes tradable symbols plus any symbols needed for filters (like VIX).
    symbols_to_load = set(symbols_to_trade)
    if USE_DYNAMIC_UNIVERSE:
        symbols_to_load.add(VIX_SYMBOL)
    
    all_symbols_to_load = sorted(list(symbols_to_load))
    
    logging.info(f"Total symbols to run backtest on: {len(symbols_to_trade)}")
    logging.info(f"Total unique symbols to load data for: {len(all_symbols_to_load)}")
    
    # --- PRE-CALCULATE DAILY FILTERS AND DATA ---
    daily_filters = {}
    daily_data_map = {}
    
    logging.info("Pre-loading all daily data for analysis...")
    tz = timezone('Asia/Kolkata')
    for symbol in all_symbols_to_load:
        daily_path = os.path.join(DATA_DIRECTORY_DAILY, f"{symbol}_daily_with_indicators.parquet")
        if os.path.exists(daily_path):
            df_daily = pd.read_parquet(daily_path)
            # --- DEFINITIVE FIX: Make daily data timezone-aware ---
            df_daily.index = df_daily.index.tz_localize(tz)
            df_daily.index = df_daily.index.normalize()
            daily_data_map[symbol] = df_daily

    if USE_DYNAMIC_UNIVERSE:
        logging.info("Pre-calculating daily filters for dynamic universe...")
        vix_df = daily_data_map.get(VIX_SYMBOL)
        if vix_df is None:
            logging.error(f"FATAL: VIX data not found for symbol '{VIX_SYMBOL}' in the 'daily' directory.")
            logging.error("Dynamic universe filter cannot run without VIX data. Exiting.")
            sys.exit(1)
        
        vix_ok_series = vix_df['close'] > VIX_THRESHOLD
        
        atr_col = f'atr_{VOLATILITY_ATR_PERIOD}_pct'
        all_volatility = {s: d[atr_col] for s, d in daily_data_map.items() if atr_col in d.columns}
        
        volatility_df = pd.DataFrame(all_volatility)
        daily_thresholds = volatility_df.quantile(VOLATILITY_UNIVERSE_PERCENTILE, axis=1)

        for date in vix_ok_series.index:
            vix_ok = vix_ok_series.get(date, False)
            hotlist = set()
            if vix_ok:
                threshold = daily_thresholds.get(date)
                if pd.notna(threshold):
                    daily_vols = volatility_df.loc[date].dropna()
                    hotlist = set(daily_vols[daily_vols > threshold].index)
            
            daily_filters[date] = {'vix_ok': vix_ok, 'hotlist': hotlist}
        logging.info("Daily filters calculated.")


    # --- PARALLEL EXECUTION ---
    num_processes = max(1, cpu_count() - 1)
    logging.info(f"Starting parallel backtest with {num_processes} workers...")

    pool = Pool(processes=num_processes)
    try:
        worker_func = partial(process_symbol_backtest, daily_filters=daily_filters, daily_data_map=daily_data_map)
        results = pool.map(worker_func, symbols_to_trade)
    except KeyboardInterrupt:
        logging.warning("\n--- Process interrupted by user (Ctrl+C). Shutting down workers. ---")
        pool.terminate()
        pool.join()
        sys.exit(1)
    finally:
        pool.close()
        pool.join()
    
    all_symbols_trades = [res for res in results if res is not None]

    if all_symbols_trades:
        for trades_df in all_symbols_trades:
            symbol_name = trades_df['symbol'].iloc[0]
            calculate_and_log_metrics(trades_df, symbol_name)

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
