# shorts_simulator.py
#
# Description:
# A dedicated backtester for analyzing and optimizing the short side of the strategy.
# ENHANCEMENT: Fixed "Merge keys contain null values" error by adding robust
# data cleaning to handle invalid timestamps in daily data files.

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

# --- UNIVERSE SELECTION ---
RUN_INDICES_ONLY = False

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

# --- TREND FILTER FOR INDICES ---
# Since indices lack volume for MVWAP, this EMA is used as the trend filter.
INDEX_EMA_PERIOD = 50

# --- ADDITIONAL FILTERS ---
# If True, adds a final check to only short stocks trading below their MVWAP.
USE_ADDITIONAL_MVWAP_FILTER = True
MVWAP_PERIOD = 50
# If True, only allows shorts if the daily RSI is below the threshold.
USE_RSI_FILTER = True
RSI_THRESHOLD = 30.0


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
    logging.info(f"  - Run Indices Only Mode: {RUN_INDICES_ONLY}")
    logging.info(f"  - Using Additional MVWAP Filter: {USE_ADDITIONAL_MVWAP_FILTER} (Period: {MVWAP_PERIOD})")
    logging.info(f"  - Using RSI Filter: {USE_RSI_FILTER} (RSI < {RSI_THRESHOLD})")
    logging.info("-" * 30)

def run_backtest(df, symbol, daily_data_map):
    """Main backtesting function that iterates through the data candle by candle."""
    trades = []
    active_trade = None
    atr_col = f'atr_{ATR_TS_PERIOD}'
    is_index = "-INDEX" in symbol
    
    symbol_daily_data = daily_data_map.get(symbol)

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
    green_blocks = (is_green != is_green.shift()).cumsum()
    consecutive_greens = is_green.groupby(green_blocks).cumsum()
    consecutive_greens[~is_green] = 0

    num_green_candles_prev = consecutive_greens.shift(1).fillna(0)
    is_signal_candle = (is_red & (num_green_candles_prev >= MIN_CONSECUTIVE_CANDLES) & (num_green_candles_prev <= MAX_PATTERN_LOOKBACK - 1))
    df['trade_trigger'] = is_signal_candle.shift(1).fillna(False)
    df['pattern_len'] = (num_green_candles_prev + 1).shift(1).fillna(0)

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
            active_trade['mfe'] = max(active_trade['mfe'], active_trade['entry_price'] - np_low[i])
            active_trade['mae'] = max(active_trade['mae'], np_high[i] - active_trade['entry_price'])

            # --- BREAKEVEN STOP LOGIC ---
            if USE_BREAKEVEN_STOP and not active_trade['be_activated']:
                if np_low[i] <= active_trade['be_trigger_price']:
                    active_trade['sl'] = active_trade['be_target_price']
                    active_trade['be_activated'] = True
            
            # --- MULTI-STAGE TRAILING STOP ACTIVATION ---
            if USE_MULTI_STAGE_TS:
                if active_trade['initial_risk'] > 0:
                    current_profit_r = (active_trade['entry_price'] - np_low[i]) / active_trade['initial_risk']
                    if current_profit_r >= AGGRESSIVE_TS_TRIGGER_R:
                        active_trade['current_ts_multiplier'] = AGGRESSIVE_TS_MULTIPLIER

            # --- ATR Trailing Stop Logic ---
            if USE_ATR_TRAILING_STOP and pd.notna(np_atr[i]):
                current_atr = np_atr[i]
                ts_multiplier = active_trade['current_ts_multiplier']
                new_trailing_stop = np_low[i] + (current_atr * ts_multiplier)
                active_trade['sl'] = min(active_trade['sl'], new_trailing_stop)

            # --- Exit Logic ---
            exit_reason, exit_price = None, None
            if np_high[i] >= active_trade['sl']: exit_reason, exit_price = 'SL_HIT', active_trade['sl']
            elif np_low[i] <= active_trade['tp']: exit_reason, exit_price = 'TP_HIT', active_trade['tp']
            
            if TRADING_MODE == 'INTRADAY' and current_timestamp.strftime('%H:%M') == EOD_TIME and not exit_reason:
                exit_reason, exit_price = 'EOD_EXIT', np_close[i]
            
            if exit_reason:
                active_trade.update({'exit_time': current_timestamp, 'exit_price': exit_price, 'exit_reason': exit_reason, 'pnl': (active_trade['entry_price'] - exit_price)})
                trades.append(active_trade)
                active_trade = None

        if not active_trade:
            if np_trade_trigger[i]:
                # --- FILTERING LOGIC (now using pre-mapped daily data) ---
                if USE_RSI_FILTER and (pd.isna(np_daily_rsi[i]) or np_daily_rsi[i] >= RSI_THRESHOLD):
                    continue

                if is_index:
                    if pd.isna(np_ema[i]) or np_close[i] > np_ema[i]:
                        continue
                else: # It's a stock
                    if USE_ADDITIONAL_MVWAP_FILTER and (pd.isna(np_mvwap[i]) or np_close[i] > np_mvwap[i]):
                        continue

                pattern_len = np_pattern_len[i]
                signal_candle_idx = i - 1
                pattern_start_idx = signal_candle_idx - pattern_len + 1
                
                pattern_high = np.max(np_high[pattern_start_idx : signal_candle_idx + 1])
                pattern_low = np.min(np_low[pattern_start_idx : signal_candle_idx + 1])

                if np_low[i] < pattern_low:
                    entry_price, sl = pattern_low, pattern_high
                    initial_risk = sl - entry_price
                    if initial_risk <= 0: continue
                    
                    tp = entry_price - (initial_risk * RISK_REWARD_RATIO)
                    active_trade = {
                        'symbol': symbol, 'direction': 'SHORT', 'entry_time': current_timestamp, 
                        'entry_price': entry_price, 'sl': sl, 'tp': tp, 'mfe': 0, 'mae': 0,
                        'initial_risk': initial_risk, 'be_activated': False,
                        'be_trigger_price': entry_price - (initial_risk * BREAKEVEN_TRIGGER_R),
                        'be_target_price': entry_price - (initial_risk * BREAKEVEN_PROFIT_R),
                        'current_ts_multiplier': ATR_TS_MULTIPLIER,
                        'daily_vol_at_entry': np_daily_vol[i],
                        'daily_rsi_at_entry': np_daily_rsi[i]
                    }
                    if np_high[i] >= sl:
                        active_trade.update({'exit_time': current_timestamp, 'exit_price': sl, 'exit_reason': 'SL_HIT_ON_ENTRY', 'pnl': entry_price - sl})
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

def process_symbol_backtest(symbol, daily_data_map):
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
        trades = run_backtest(df_filtered, symbol, daily_data_map)
        
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
        worker_func = partial(process_symbol_backtest, daily_data_map=daily_data_map)
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
