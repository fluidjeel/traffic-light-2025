# longs_simulator.py
#
# Description:
# A dedicated backtester for analyzing and optimizing the long side of the strategy.

import pandas as pd
import os
import datetime
import logging
import sys
import numpy as np
from pytz import timezone
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# -- PROJECT ROOT DIRECTORY --
# Determines the root of the project to build all other file paths from.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# -- DATE RANGE FOR BACKTEST --
# The start date for the backtest in 'YYYY-MM-DD' format.
START_DATE = '2018-01-01'
# The end date for the backtest in 'YYYY-MM-DD' format.
END_DATE = '2025-01-31'

# -- STRATEGY CONFIGURATION --
# A unique name for the strategy run, used for creating log directories.
STRATEGY_NAME = "TrafficLight-Manny-LONGS_ONLY"
# The desired risk-reward ratio for setting the take-profit target.
RISK_REWARD_RATIO = 10.0

# --- DYNAMIC PATTERN CONFIGURATION ---
# The maximum number of consecutive red candles to consider for the pattern.
MAX_PATTERN_LOOKBACK = 10
# The minimum number of consecutive red candles required for a valid pattern.
MIN_CONSECUTIVE_CANDLES = 1

# --- UNIVERSE SELECTION ---
# If True, the backtest will only run on the specified index symbols.
RUN_INDICES_ONLY = False

# -- SIMULATOR MODE --
# "INTRADAY" for trades that are closed by EOD, "POSITIONAL" for trades that can be held overnight.
TRADING_MODE = "POSITIONAL"

# -- TRADE MANAGEMENT --
# If True, an ATR-based trailing stop will be used.
USE_ATR_TRAILING_STOP = True
# The period for calculating the Average True Range (ATR).
ATR_TS_PERIOD = 14
# The multiplier for the ATR to set the trailing stop distance.
# Using 2.5 based on learnings from the shorts simulator.
ATR_TS_MULTIPLIER = 4.0

# -- BREAKEVEN STOP LOGIC --
# If True, a breakeven stop will be used.
USE_BREAKEVEN_STOP = True
# The multiple of initial risk at which the breakeven stop is triggered.
BREAKEVEN_TRIGGER_R = 1.0
# The multiple of initial risk at which the breakeven stop is placed (locking in profit).
BREAKEVEN_PROFIT_R = 0.1

# -- MULTI-STAGE TRAILING STOP --
# If True, a more aggressive trailing stop will be activated after a certain profit.
USE_MULTI_STAGE_TS = True
# The multiple of initial risk that triggers the aggressive trailing stop.
AGGRESSIVE_TS_TRIGGER_R = 5.0
# The new, more aggressive multiplier for the trailing stop.
AGGRESSIVE_TS_MULTIPLIER = 1.0

# --- EXPERIMENTAL FILTERS ---
# If True, a new multi-tiered profit lock stop will be used to protect profits.
USE_PROFIT_LOCK_STOP = False
# The multiple of initial risk at which the profit lock stop is triggered.
PROFIT_LOCK_TRIGGER_R = 1.5
# The multiple of initial risk at which the profit lock stop is placed.
PROFIT_LOCK_PROFIT_R = 0.5

# --- TREND FILTER FOR INDICES ---
# Since indices lack volume for MVWAP, this EMA is used as the trend filter.
INDEX_EMA_PERIOD = 50

# --- ADDITIONAL FILTERS ---
# If True, adds a final check to only long stocks trading above their MVWAP.
USE_ADDITIONAL_MVWAP_FILTER = True
# The period for the Moving Volume Weighted Average Price (MVWAP) filter.
MVWAP_PERIOD = 50
# If True, only allows longs if the daily RSI is above the threshold.
USE_RSI_FILTER = True
# The RSI threshold for the daily RSI filter. Using 70.0 as a symmetrical starting point.
RSI_THRESHOLD = 75.0

# -- DATA & FILE PATHS --
# Directory containing the 15-minute processed data files.
DATA_DIRECTORY_15MIN = os.path.join(ROOT_DIR, "data", "universal_processed", "15min")
# Directory containing the daily processed data files.
DATA_DIRECTORY_DAILY = os.path.join(ROOT_DIR, "data", "universal_processed", "daily")
# Path to the CSV file containing the list of symbols to backtest.
SYMBOL_LIST_PATH = os.path.join(ROOT_DIR, "nifty200_fno.csv")

# -- ADDITIONAL SYMBOLS --
# A list of additional symbols, such as indices, to include in the backtest.
ADDITIONAL_SYMBOLS = ['NIFTY50-INDEX', 'NIFTYBANK-INDEX']

# -- LOGGING CONFIGURATION --
# Base directory where all backtest logs will be stored.
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")

# -- TRADING SESSION --
# The time of day for End-of-Day (EOD) exit in 'HH:MM' format.
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
    logging.info("--- Starting LONGS-ONLY Backtest Simulation ---")
    
    # Dynamically log all configuration variables defined in this section.
    config_vars = {
        'Strategy Name': STRATEGY_NAME,
        'Backtest Period': f"{START_DATE} to {END_DATE}",
        'Trading Mode': TRADING_MODE,
        'Run Indices Only Mode': RUN_INDICES_ONLY,
        'Risk/Reward Ratio': RISK_REWARD_RATIO,
        'Pattern Lookback': f"Min: {MIN_CONSECUTIVE_CANDLES}, Max: {MAX_PATTERN_LOOKBACK-1}",
        'Using Additional MVWAP Filter': f"{USE_ADDITIONAL_MVWAP_FILTER} (Period: {MVWAP_PERIOD})",
        'Using RSI Filter': f"{USE_RSI_FILTER} (RSI > {RSI_THRESHOLD})",
        'Using ATR Trailing Stop': f"{USE_ATR_TRAILING_STOP} (Period: {ATR_TS_PERIOD}, Multiplier: {ATR_TS_MULTIPLIER})",
        'Using Breakeven Stop': f"{USE_BREAKEVEN_STOP} (Trigger: {BREAKEVEN_TRIGGER_R}R, Profit: {BREAKEVEN_PROFIT_R}R)",
        'Using Multi-Stage Trailing Stop': f"{USE_MULTI_STAGE_TS} (Trigger: {AGGRESSIVE_TS_TRIGGER_R}R, Multiplier: {AGGRESSIVE_TS_MULTIPLIER})",
        'Using Profit Lock Stop': f"{USE_PROFIT_LOCK_STOP} (Trigger: {PROFIT_LOCK_TRIGGER_R}R, Profit: {PROFIT_LOCK_PROFIT_R}R)",
    }
    
    for key, value in config_vars.items():
        logging.info(f"  - {key}: {value}")

    logging.info("-" * 30)

def run_backtest(df, symbol, symbol_daily_data):
    """Main backtesting function that iterates through the data candle by candle."""
    trades = []
    active_trade = None
    atr_col = f'atr_{ATR_TS_PERIOD}'
    is_index = "-INDEX" in symbol
    
    if df.empty:
        return trades

    # --- OPTIMIZATION: Map daily data to 15-min frame before the loop ---
    if symbol_daily_data is not None and not symbol_daily_data.empty:
        daily_filters_to_map = symbol_daily_data[[f'atr_{ATR_TS_PERIOD}_pct', 'rsi_14']].copy()
        daily_filters_to_map.rename(columns={
            f'atr_{ATR_TS_PERIOD}_pct': 'daily_vol_pct',
            'rsi_14': 'daily_rsi'
        }, inplace=True)
        # Use merge_asof to efficiently map the daily value to all of the day's 15-min candles
        df = pd.merge_asof(df.sort_index(), daily_filters_to_map.sort_index(), left_index=True, right_index=True, direction='backward')

    # --- Vectorized Trade Setup Identification ---
    is_red = df['close'] < df['open']
    is_green = df['close'] > df['open']
    red_blocks = (is_red != is_red.shift()).cumsum()
    consecutive_reds = is_red.groupby(red_blocks).cumsum()
    consecutive_reds[~is_red] = 0

    num_red_candles_prev = consecutive_reds.shift(1).fillna(0)
    is_signal_candle = (is_green & (num_red_candles_prev >= MIN_CONSECUTIVE_CANDLES) & (num_red_candles_prev <= MAX_PATTERN_LOOKBACK - 1))
    df['trade_trigger'] = is_signal_candle.shift(1).fillna(False)
    df['pattern_len'] = (num_red_candles_prev + 1).shift(1).fillna(0)

    # --- Convert columns to NumPy arrays for faster loop access ---
    np_high = df['high'].to_numpy()
    np_low = df['low'].to_numpy()
    np_close = df['close'].to_numpy()
    np_trade_trigger = df['trade_trigger'].to_numpy()
    np_pattern_len = df['pattern_len'].to_numpy(dtype=np.int32)
    np_atr = df[atr_col].to_numpy() if atr_col in df.columns else np.full(len(df), np.nan)
    mvwap_col = f'mvwap_{MVWAP_PERIOD}'
    np_mvwap = df[mvwap_col].to_numpy() if mvwap_col in df.columns else np.full(len(df), np.nan)
    ema_col = f'ema_{INDEX_EMA_PERIOD}'
    np_ema = df[ema_col].to_numpy() if ema_col in df.columns else np.full(len(df), np.nan)
    np_daily_rsi = df['daily_rsi'].to_numpy() if 'daily_rsi' in df.columns else np.full(len(df), np.nan)
    np_daily_vol = df['daily_vol_pct'].to_numpy() if 'daily_vol_pct' in df.columns else np.full(len(df), np.nan)
    df_index = df.index

    for i in range(1, len(df)):
        current_timestamp = df_index[i]
        current_date = current_timestamp.normalize()

        if active_trade:
            # --- MFE and MAE Tracking ---
            active_trade['mfe'] = max(active_trade['mfe'], np_high[i] - active_trade['entry_price'])
            active_trade['mae'] = max(active_trade['mae'], active_trade['entry_price'] - np_low[i])

            # --- DYNAMIC TRADE MANAGEMENT LOGIC ---
            current_profit_r = (np_high[i] - active_trade['entry_price']) / active_trade['initial_risk']

            # 1. BREAKEVEN STOP LOGIC (Existing)
            if USE_BREAKEVEN_STOP and not active_trade['be_activated']:
                if np_high[i] >= active_trade['be_trigger_price']:
                    active_trade['sl'] = active_trade['be_target_price']
                    active_trade['be_activated'] = True
            
            # 2. PROFIT LOCK STOP LOGIC (Existing)
            if USE_PROFIT_LOCK_STOP and not active_trade.get('pl_activated', False):
                if current_profit_r >= PROFIT_LOCK_TRIGGER_R:
                    active_trade['sl'] = active_trade['entry_price'] + (active_trade['initial_risk'] * PROFIT_LOCK_PROFIT_R)
                    active_trade['pl_activated'] = True
            
            # 3. MULTI-STAGE TRAILING STOP ACTIVATION (Existing)
            if USE_MULTI_STAGE_TS:
                if active_trade['initial_risk'] > 0:
                    if current_profit_r >= AGGRESSIVE_TS_TRIGGER_R:
                        active_trade['current_ts_multiplier'] = AGGRESSIVE_TS_MULTIPLIER

            # 4. ATR Trailing Stop Logic (Existing)
            if USE_ATR_TRAILING_STOP and pd.notna(np_atr[i]):
                current_atr = np_atr[i]
                ts_multiplier = active_trade['current_ts_multiplier']
                new_trailing_stop = np_high[i] - (current_atr * ts_multiplier)
                active_trade['sl'] = max(active_trade['sl'], new_trailing_stop)

            # --- Exit Logic ---
            exit_reason, exit_price = None, None
            initial_sl_price = active_trade['entry_price'] - active_trade['initial_risk']
            
            if np_low[i] <= active_trade['sl']:
                exit_price = active_trade['sl']
                if active_trade.get('pl_activated', False):
                    exit_reason = 'PROFIT_LOCK_HIT'
                elif active_trade['be_activated']:
                    exit_reason = 'BE_HIT'
                elif active_trade['sl'] != initial_sl_price:
                    exit_reason = 'TS_HIT'
                else:
                    exit_reason = 'SL_HIT'
            elif np_high[i] >= active_trade['tp']:
                exit_reason, exit_price = 'TP_HIT', active_trade['tp']
            
            if TRADING_MODE == 'INTRADAY' and current_timestamp.strftime('%H:%M') == EOD_TIME and not exit_reason:
                exit_reason, exit_price = 'EOD_EXIT', np_close[i]
            
            if exit_reason:
                active_trade.update({'exit_time': current_timestamp, 'exit_price': exit_price, 'exit_reason': exit_reason, 'pnl': (exit_price - active_trade['entry_price'])})
                trades.append(active_trade)
                active_trade = None

        if not active_trade:
            if np_trade_trigger[i]:
                # --- FILTERING LOGIC (now using pre-mapped daily data) ---
                if USE_RSI_FILTER and (pd.isna(np_daily_rsi[i]) or np_daily_rsi[i] <= RSI_THRESHOLD):
                    continue

                if is_index:
                    if pd.isna(np_ema[i]) or np_close[i] < np_ema[i]:
                        continue
                else: # It's a stock
                    if USE_ADDITIONAL_MVWAP_FILTER and (pd.isna(np_mvwap[i]) or np_close[i] < np_mvwap[i]):
                        continue

                pattern_len = np_pattern_len[i]
                signal_candle_idx = i - 1
                pattern_start_idx = signal_candle_idx - pattern_len + 1
                
                pattern_high = np.max(np_high[pattern_start_idx : signal_candle_idx + 1])
                pattern_low = np.min(np_low[pattern_start_idx : signal_candle_idx + 1])

                if np_high[i] > pattern_high:
                    entry_price, sl = pattern_high, pattern_low
                    initial_risk = entry_price - sl
                    if initial_risk <= 0: continue
                    
                    tp = entry_price + (initial_risk * RISK_REWARD_RATIO)
                    active_trade = {
                        'symbol': symbol, 'direction': 'LONG', 'entry_time': current_timestamp, 
                        'entry_price': entry_price, 'sl': sl, 'tp': tp, 'mfe': 0, 'mae': 0,
                        'initial_risk': initial_risk, 'be_activated': False,
                        'pl_activated': False,
                        'be_trigger_price': entry_price + (initial_risk * BREAKEVEN_TRIGGER_R),
                        'be_target_price': entry_price + (initial_risk * BREAKEVEN_PROFIT_R),
                        'current_ts_multiplier': ATR_TS_MULTIPLIER,
                        'daily_vol_at_entry': np_daily_vol[i],
                        'daily_rsi_at_entry': np_daily_rsi[i]
                    }
                    if np_low[i] <= sl:
                        active_trade.update({'exit_time': current_timestamp, 'exit_price': sl, 'exit_reason': 'SL_HIT_ON_ENTRY', 'pnl': sl - entry_price})
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

def process_symbol_backtest(symbol, symbol_daily_data):
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
        trades = run_backtest(df_filtered, symbol, symbol_daily_data)
        
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

    if RUN_INDICES_ONLY:
        symbols_to_trade = sorted(ADDITIONAL_SYMBOLS)
    else:
        try:
            symbols_df = pd.read_csv(SYMBOL_LIST_PATH)
            symbols_from_csv = symbols_df['symbol'].tolist()
            symbols_to_trade = sorted(list(set(symbols_from_csv + ADDITIONAL_SYMBOLS)))
        except FileNotFoundError:
            logging.warning(f"Symbol list file not found at: {SYMBOL_LIST_PATH}. Running on indices only.")
            symbols_to_trade = sorted(ADDITIONAL_SYMBOLS)
    
    logging.info(f"Total symbols to run backtest on: {len(symbols_to_trade)}")
    
    # --- PRE-CALCULATE DAILY DATA ---
    daily_data_map = {}
    
    logging.info("Pre-loading all daily data for analysis...")
    tz = timezone('Asia/Kolkata')
    for symbol in symbols_to_trade:
        daily_path = os.path.join(DATA_DIRECTORY_DAILY, f"{symbol}_daily_with_indicators.parquet")
        if os.path.exists(daily_path):
            df_daily = pd.read_parquet(daily_path)
            # --- FIX: Drop rows with invalid (NaT) timestamps in the index ---
            df_daily = df_daily[df_daily.index.notna()]
            if not df_daily.empty:
                df_daily.index = df_daily.index.tz_localize(tz)
                df_daily.index = df_daily.index.normalize()
                daily_data_map[symbol] = df_daily

    # --- PARALLEL EXECUTION ---
    num_processes = max(1, cpu_count() - 1)
    logging.info(f"Starting parallel backtest with {num_processes} workers...")

    pool = Pool(processes=num_processes)
    try:
        # Prepare arguments for starmap. This is more efficient than passing the
        # entire daily_data_map to each worker via partial. Each worker now
        # only receives the daily data for the one symbol it is processing.
        args_for_workers = [(symbol, daily_data_map.get(symbol)) for symbol in symbols_to_trade]
        # Use starmap to pass the tuple of arguments to each worker.
        results = pool.starmap(process_symbol_backtest, args_for_workers)
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
