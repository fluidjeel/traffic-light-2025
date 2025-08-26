# main_tfl_simulator.py
#
# Description:
# A comprehensive backtester for the combined "all-weather" TrafficLight-Manny strategy,
# incorporating dynamic position sizing, portfolio-level risk management, and market realism.

import pandas as pd
import os
import datetime
import logging
import sys
import numpy as np
from pytz import timezone
from collections import defaultdict
import warnings

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# -- GLOBAL CONFIGS --
# Determines the root of the project to build all other file paths from.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# The start date for the backtest in 'YYYY-MM-DD' format.
START_DATE = '2018-01-01'
# The end date for the backtest in 'YYYY-MM-DD' format.
END_DATE = '2025-01-31'
# A unique name for the strategy run, used for creating log directories.
STRATEGY_NAME = "TrafficLight-Manny-ALL_WEATHER"
# The initial capital for the simulation in INR.
INITIAL_CAPITAL = 1000000.0
# The percentage of capital to risk per trade. (e.g., 1.0 = 1%)
RISK_PER_TRADE_PCT = 1.0
# The maximum total risk allowed for all open positions. (e.g., 4.0 = 4%)
MAX_PORTFOLIO_RISK_PCT = 4.0
# Slippage in basis points (e.g., 2.0 = 0.02% of trade value). Applied on entry and exit.
SLIPPAGE_BPS = 2.0

# -- LONGS CONFIGS --
LONGS_RSI_THRESHOLD = 75.0
LONGS_ATR_TS_MULTIPLIER = 4.0
LONGS_BREAKEVEN_TRIGGER_R = 1.0
LONGS_BREAKEVEN_PROFIT_R = 0.1
LONGS_AGGRESSIVE_TS_TRIGGER_R = 5.0
LONGS_AGGRESSIVE_TS_MULTIPLIER = 1.0

# -- SHORTS CONFIGS --
SHORTS_RSI_THRESHOLD = 25.0
SHORTS_ATR_TS_MULTIPLIER = 2.5
SHORTS_BREAKEVEN_TRIGGER_R = 1.0
SHORTS_BREAKEVEN_PROFIT_R = 0.1
SHORTS_AGGRESSIVE_TS_TRIGGER_R = 3.0
SHORTS_AGGRESSIVE_TS_MULTIPLIER = 1.0

# --- SHARED FILTERS & SETTINGS ---
RISK_REWARD_RATIO = 10.0
MAX_PATTERN_LOOKBACK = 10
MIN_CONSECUTIVE_CANDLES = 1
USE_ADDITIONAL_MVWAP_FILTER = True
MVWAP_PERIOD = 50
USE_RSI_FILTER = True
INDEX_EMA_PERIOD = 50
TRADING_MODE = "POSITIONAL"
EOD_TIME = "15:15"
ADDITIONAL_SYMBOLS = ['NIFTY50-INDEX', 'NIFTYBANK-INDEX']

# -- DATA & FILE PATHS --
DATA_DIRECTORY_15MIN = os.path.join(ROOT_DIR, "data", "universal_processed", "15min")
DATA_DIRECTORY_DAILY = os.path.join(ROOT_DIR, "data", "universal_processed", "daily")
SYMBOL_LIST_PATH = os.path.join(ROOT_DIR, "nifty200_fno.csv")
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")


# ==============================================================================
# --- BACKTESTING ENGINE ---
# ==============================================================================

class PortfolioManager:
    """Manages the overall portfolio state, including capital, open positions, and risk."""
    def __init__(self, initial_capital, risk_per_trade_pct, max_portfolio_risk_pct):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade_value = initial_capital * (risk_per_trade_pct / 100)
        self.max_portfolio_risk_value = initial_capital * (max_portfolio_risk_pct / 100)
        self.active_trades = []
        self.total_risk_at_entry = 0.0

    def check_and_add_trade(self, trade):
        """Checks if a new trade can be opened without exceeding max portfolio risk."""
        new_total_risk = self.total_risk_at_entry + trade['risk_value']
        if new_total_risk > self.max_portfolio_risk_value:
            return False, "Max portfolio risk exceeded."
        if trade['cost'] > self.capital:
            return False, "Insufficient capital."
        
        self.active_trades.append(trade)
        self.capital -= trade['cost']
        self.total_risk_at_entry = new_total_risk
        return True, None

    def update_on_exit(self, trade, pnl):
        """Updates portfolio state when a trade is closed."""
        self.capital += trade['cost'] + pnl
        self.total_risk_at_entry -= trade['risk_value']
        self.active_trades.remove(trade)


def setup_logging(log_dir):
    """Configures the logging for the simulation summary."""
    summary_file_path = os.path.join(log_dir, 'summary.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(summary_file_path), logging.StreamHandler()])

def log_configs():
    """Logs the configuration settings used for the backtest run."""
    logging.info("--- Starting ALL_WEATHER Backtest Simulation ---")
    
    config_vars = {
        'Strategy Name': STRATEGY_NAME,
        'Backtest Period': f"{START_DATE} to {END_DATE}",
        'Initial Capital': f"INR {INITIAL_CAPITAL:,.2f}",
        'Risk per Trade': f"{RISK_PER_TRADE_PCT}% (INR {INITIAL_CAPITAL * RISK_PER_TRADE_PCT / 100:,.2f})",
        'Max Portfolio Risk': f"{MAX_PORTFOLIO_RISK_PCT}% (INR {INITIAL_CAPITAL * MAX_PORTFOLIO_RISK_PCT / 100:,.2f})",
        'Slippage': f"{SLIPPAGE_BPS} bps",
        'Longs | RSI Filter': f"RSI > {LONGS_RSI_THRESHOLD}",
        'Longs | ATR Multiplier': LONGS_ATR_TS_MULTIPLIER,
        'Shorts | RSI Filter': f"RSI < {SHORTS_RSI_THRESHOLD}",
        'Shorts | ATR Multiplier': SHORTS_ATR_TS_MULTIPLIER,
    }
    
    for key, value in config_vars.items():
        logging.info(f"  - {key}: {value}")

    logging.info("-" * 30)

def calculate_position_size(risk_value, initial_risk_per_unit):
    """Calculates the number of units to trade based on a fixed risk amount."""
    if initial_risk_per_unit <= 0:
        return 0, 0
    num_units = risk_value / initial_risk_per_unit
    return int(num_units), num_units * initial_risk_per_unit

def run_simulation(all_data_map, portfolio, trade_log, rejected_log, signals_by_timestamp):
    """
    Main simulation loop that processes all symbols chronologically.
    """
    tz = timezone('Asia/Kolkata')
    
    # Get all unique timestamps from all symbols
    all_timestamps = sorted(list(set(ts for df in all_data_map.values() for ts in df.index)))
    
    active_trades = []
    
    for current_timestamp in all_timestamps:
        current_date = current_timestamp.normalize()
        
        # --- CHECK AND MANAGE OPEN TRADES FIRST ---
        trades_to_remove = []
        for trade in active_trades:
            symbol = trade['symbol']
            
            # Find the candle data for this timestamp
            df_symbol = all_data_map.get(symbol)
            if df_symbol is None or current_timestamp not in df_symbol.index:
                continue

            current_candle = df_symbol.loc[current_timestamp]
            np_high = current_candle['high']
            np_low = current_candle['low']
            np_close = current_candle['close']

            # --- GAP HANDLING ---
            # Added a check to prevent KeyError when the index is at the beginning of the DataFrame
            if current_timestamp > df_symbol.index[0] and current_candle.name.date() != df_symbol.index[df_symbol.index.get_loc(current_timestamp) - 1].date():
                if trade['direction'] == 'LONG' and np_low <= trade['sl']:
                    exit_price = min(trade['sl'], np_low) * (1 - SLIPPAGE_BPS / 10000)
                    pnl = (exit_price - trade['entry_price']) * trade['num_units']
                    trade_log.append(trade)
                    trades_to_remove.append(trade)
                    continue
                elif trade['direction'] == 'SHORT' and np_high >= trade['sl']:
                    exit_price = max(trade['sl'], np_high) * (1 + SLIPPAGE_BPS / 10000)
                    pnl = (trade['entry_price'] - exit_price) * trade['num_units']
                    trade_log.append(trade)
                    trades_to_remove.append(trade)
                    continue

            # --- DYNAMIC TRADE MANAGEMENT LOGIC ---
            initial_risk = trade['initial_risk']
            current_profit_r = (np_high - trade['entry_price']) / initial_risk if trade['direction'] == 'LONG' else (trade['entry_price'] - np_low) / initial_risk

            # 1. BREAKEVEN STOP
            if trade['direction'] == 'LONG' and not trade['be_activated'] and np_high >= trade['be_trigger_price']:
                trade['sl'] = trade['be_target_price']
                trade['be_activated'] = True
            elif trade['direction'] == 'SHORT' and not trade['be_activated'] and np_low <= trade['be_trigger_price']:
                trade['sl'] = trade['be_target_price']
                trade['be_activated'] = True
            
            # 2. MULTI-STAGE TRAILING STOP
            if trade['initial_risk'] > 0 and trade['current_ts_multiplier'] != trade['aggressive_ts_multiplier'] and current_profit_r >= trade['aggressive_ts_trigger_r']:
                trade['current_ts_multiplier'] = trade['aggressive_ts_multiplier']

            # 3. ATR Trailing Stop
            atr_col = f'atr_{trade["atr_ts_period"]}'
            if atr_col in current_candle and pd.notna(current_candle[atr_col]):
                current_atr = current_candle[atr_col]
                ts_multiplier = trade['current_ts_multiplier']
                
                if trade['direction'] == 'LONG':
                    new_trailing_stop = np_high - (current_atr * ts_multiplier)
                    trade['sl'] = max(trade['sl'], new_trailing_stop)
                else: # SHORT
                    new_trailing_stop = np_low + (current_atr * ts_multiplier)
                    trade['sl'] = min(trade['sl'], new_trailing_stop)

            # --- CHECK FOR EXIT CONDITIONS ---
            exit_reason, exit_price = None, None
            if trade['direction'] == 'LONG' and np_low <= trade['sl']:
                exit_price = trade['sl'] * (1 - SLIPPAGE_BPS / 10000)
                if trade['sl'] == trade['be_target_price']: exit_reason = 'BE_HIT'
                else: exit_reason = 'TS_HIT'
            elif trade['direction'] == 'LONG' and np_high >= trade['tp']:
                exit_price = trade['tp'] * (1 - SLIPPAGE_BPS / 10000)
                exit_reason = 'TP_HIT'
            elif trade['direction'] == 'SHORT' and np_high >= trade['sl']:
                exit_price = trade['sl'] * (1 + SLIPPAGE_BPS / 10000)
                if trade['sl'] == trade['be_target_price']: exit_reason = 'BE_HIT'
                else: exit_reason = 'TS_HIT'
            elif trade['direction'] == 'SHORT' and np_low <= trade['tp']:
                exit_price = trade['tp'] * (1 + SLIPPAGE_BPS / 10000)
                exit_reason = 'TP_HIT'

            # EOD exit for intraday mode
            if TRADING_MODE == 'INTRADAY' and current_timestamp.strftime('%H:%M') == EOD_TIME and not exit_reason:
                exit_reason, exit_price = 'EOD_EXIT', np_close
            
            if exit_reason:
                trade['exit_time'] = current_timestamp
                trade['exit_price'] = exit_price
                trade['exit_reason'] = exit_reason
                trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['num_units'] if trade['direction'] == 'LONG' else (trade['entry_price'] - trade['exit_price']) * trade['num_units']
                trade_log.append(trade)
                trades_to_remove.append(trade)
        
        # Remove exited trades from active list
        for trade in trades_to_remove:
            portfolio.update_on_exit(trade, trade['pnl'])

        # --- LOOK FOR NEW SIGNALS (OPTIMIZED) ---
        # Instead of looping through all symbols, we do a quick lookup
        # in our pre-computed map of signals for the current timestamp.
        if current_timestamp in signals_by_timestamp:
            active_symbols = {t['symbol'] for t in active_trades} # Fast lookup set
            for signal in signals_by_timestamp[current_timestamp]:
                symbol = signal['symbol']
                if symbol in active_symbols:
                    continue
                
                # Check for the existence of the current candle for this symbol
                df_symbol = all_data_map.get(symbol)
                if df_symbol is None or current_timestamp not in df_symbol.index:
                    continue
                
                current_candle = df_symbol.loc[current_timestamp]
            
                # --- Check for Long Signal ---
                if signal['direction'] == 'LONG':
                    if current_candle['daily_rsi'] > LONGS_RSI_THRESHOLD and current_candle['close'] > current_candle['daily_ema_50']:
                        pattern_high = current_candle['pattern_high_long']
                        pattern_low = current_candle['pattern_low_long']
                        if current_candle['high'] > pattern_high:
                            entry_price = pattern_high * (1 + SLIPPAGE_BPS / 10000)
                            sl = pattern_low
                            initial_risk_per_unit = entry_price - sl
                            if initial_risk_per_unit <= 0: continue
                            
                            num_units, risk_value = calculate_position_size(portfolio.risk_per_trade_value, initial_risk_per_unit)
                            if num_units > 0:
                                trade = {
                                    'symbol': symbol, 'direction': 'LONG', 'entry_time': current_timestamp, 
                                    'entry_price': entry_price, 'sl': sl, 'tp': entry_price + (initial_risk_per_unit * RISK_REWARD_RATIO),
                                    'initial_risk': initial_risk_per_unit, 'be_activated': False, 'be_trigger_price': entry_price + (initial_risk_per_unit * LONGS_BREAKEVEN_TRIGGER_R),
                                    'be_target_price': entry_price + (initial_risk_per_unit * LONGS_BREAKEVEN_PROFIT_R),
                                    'aggressive_ts_trigger_r': LONGS_AGGRESSIVE_TS_TRIGGER_R, 'aggressive_ts_multiplier': LONGS_AGGRESSIVE_TS_MULTIPLIER,
                                    'atr_ts_period': 14, 'current_ts_multiplier': LONGS_ATR_TS_MULTIPLIER, 'num_units': num_units, 'risk_value': risk_value,
                                    'cost': entry_price * num_units
                                }
                                is_added, reason = portfolio.check_and_add_trade(trade)
                                if is_added: active_trades.append(trade)
                                else: rejected_log.append(trade)

                # --- Check for Short Signal ---
                if signal['direction'] == 'SHORT':
                    if current_candle['daily_rsi'] < SHORTS_RSI_THRESHOLD and current_candle['close'] < current_candle['daily_ema_50']:
                        pattern_high = current_candle['pattern_high_short']
                        pattern_low = current_candle['pattern_low_short']
                        if current_candle['low'] < pattern_low:
                            entry_price = pattern_low * (1 - SLIPPAGE_BPS / 10000)
                            sl = pattern_high
                            initial_risk_per_unit = sl - entry_price
                            if initial_risk_per_unit <= 0: continue
                            
                            num_units, risk_value = calculate_position_size(portfolio.risk_per_trade_value, initial_risk_per_unit)
                            if num_units > 0:
                                trade = {
                                    'symbol': symbol, 'direction': 'SHORT', 'entry_time': current_timestamp, 
                                    'entry_price': entry_price, 'sl': sl, 'tp': entry_price - (initial_risk_per_unit * RISK_REWARD_RATIO),
                                    'initial_risk': initial_risk_per_unit, 'be_activated': False, 'be_trigger_price': entry_price - (initial_risk_per_unit * SHORTS_BREAKEVEN_TRIGGER_R),
                                    'be_target_price': entry_price - (initial_risk_per_unit * SHORTS_BREAKEVEN_PROFIT_R),
                                    'aggressive_ts_trigger_r': SHORTS_AGGRESSIVE_TS_TRIGGER_R, 'aggressive_ts_multiplier': SHORTS_AGGRESSIVE_TS_MULTIPLIER,
                                    'atr_ts_period': 14, 'current_ts_multiplier': SHORTS_ATR_TS_MULTIPLIER, 'num_units': num_units, 'risk_value': risk_value,
                                    'cost': entry_price * num_units
                                }
                                is_added, reason = portfolio.check_and_add_trade(trade)
                                if is_added: active_trades.append(trade)
                                else: rejected_log.append(trade)


def calculate_and_log_metrics(all_trades_df, scope_name, initial_capital, log_dir):
    """Calculates and logs comprehensive performance metrics."""
    if all_trades_df.empty:
        logging.info(f"No trades executed for {scope_name}.")
        return

    # Basic Metrics
    total_trades = len(all_trades_df)
    wins = all_trades_df[all_trades_df['pnl'] > 0]
    losses = all_trades_df[all_trades_df['pnl'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = wins['pnl'].sum()
    gross_loss = losses['pnl'].sum()
    net_pnl = all_trades_df['pnl'].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
    
    # Portfolio Metrics (Simplified)
    all_trades_df.set_index('exit_time', inplace=True)
    all_trades_df.sort_index(inplace=True)
    equity_curve = initial_capital + all_trades_df['pnl'].cumsum()
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    
    # Log the summary
    logging.info(f"\n--- Performance Metrics for {scope_name} ---")
    logging.info(f"Total Trades: {total_trades}")
    logging.info(f"Win Rate: {win_rate:.2f}%")
    logging.info(f"Profit Factor: {profit_factor:.2f}")
    logging.info(f"Net PnL: {net_pnl:,.2f}")
    logging.info(f"Max Drawdown: {max_drawdown:.2%}")
    logging.info(f"Final Capital: {equity_curve.iloc[-1]:,.2f}")

    # Save metrics to a file if needed
    metrics_file = os.path.join(log_dir, 'metrics.txt')
    with open(metrics_file, 'a') as f:
        f.write(f"\n--- {scope_name} ---\n")
        f.write(f"Total Trades: {total_trades}\n")
        f.write(f"Win Rate: {win_rate:.2f}%\n")
        f.write(f"Profit Factor: {profit_factor:.2f}\n")
        f.write(f"Net PnL: {net_pnl:,.2f}\n")
        f.write(f"Max Drawdown: {max_drawdown:.2%}\n")
        f.write(f"Final Capital: {equity_curve.iloc[-1]:,.2f}\n")


def main():
    """Main function to orchestrate the backtesting process."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(LOGS_BASE_DIR, STRATEGY_NAME, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)
    log_configs()

    # --- LOAD SYMBOLS & DATA ---
    symbols_to_trade = []
    try:
        symbols_df = pd.read_csv(SYMBOL_LIST_PATH)
        symbols_from_csv = symbols_df['symbol'].tolist()
        symbols_to_trade = sorted(list(set(symbols_from_csv + ADDITIONAL_SYMBOLS)))
    except FileNotFoundError:
        logging.warning(f"Symbol list file not found. Running on indices only.")
        symbols_to_trade = sorted(ADDITIONAL_SYMBOLS)
    
    logging.info(f"Total symbols in universe: {len(symbols_to_trade)}")
    
    all_data_map = {}
    logging.info("Pre-loading and processing all data...")
    all_timestamps = set()
    signals_by_timestamp = defaultdict(list)
    
    for symbol in symbols_to_trade:
        daily_path = os.path.join(DATA_DIRECTORY_DAILY, f"{symbol}_daily_with_indicators.parquet")
        min_path = os.path.join(DATA_DIRECTORY_15MIN, f"{symbol}_15min_with_indicators.parquet")
        
        if not os.path.exists(daily_path) or not os.path.exists(min_path):
            continue

        try:
            df_daily = pd.read_parquet(daily_path)
            df_15min = pd.read_parquet(min_path)

            # --- Data Cleaning (Crucial Fix for 'Merge keys contain null values') ---
            df_daily = df_daily.loc[df_daily.index.notna()].copy()
            df_15min = df_15min.loc[df_15min.index.notna()].copy()
            
            # --- Merge Daily Data into 15-Min Dataframe for Filters ---
            df_daily.index = pd.to_datetime(df_daily.index)
            df_15min.index = pd.to_datetime(df_15min.index)
            df_daily = df_daily.rename(columns={'rsi_14': 'daily_rsi'})
            
            # Use merge_asof to align daily data with 15-minute data without lookahead bias
            df_15min = pd.merge_asof(
                df_15min.sort_index(),
                df_daily[['daily_rsi']],
                left_index=True,
                right_index=True,
                direction='backward'
            )
            df_15min = pd.merge_asof(
                df_15min.sort_index(),
                df_daily[['ema_50']].rename(columns={'ema_50': 'daily_ema_50'}),
                left_index=True,
                right_index=True,
                direction='backward'
            )

            # Pre-calculate signals
            is_red = df_15min['close'] < df_15min['open']
            is_green = df_15min['close'] > df_15min['open']

            # Long pattern
            red_blocks = (is_red != is_red.shift()).cumsum()
            consecutive_reds = is_red.groupby(red_blocks).cumsum()
            consecutive_reds[~is_red] = 0
            num_red_candles_prev = consecutive_reds.shift(1).fillna(0)
            is_signal_candle_long = (is_green & (num_red_candles_prev >= MIN_CONSECUTIVE_CANDLES) & (num_red_candles_prev <= MAX_PATTERN_LOOKBACK - 1))
            df_15min['trade_trigger_long'] = is_signal_candle_long.shift(1).fillna(False)
            df_15min['pattern_len_long'] = (num_red_candles_prev + 1).shift(1).fillna(0)
            
            # Short pattern
            green_blocks = (is_green != is_green.shift()).cumsum()
            consecutive_greens = is_green.groupby(green_blocks).cumsum()
            consecutive_greens[~is_green] = 0
            num_green_candles_prev = consecutive_greens.shift(1).fillna(0)
            is_signal_candle_short = (is_red & (num_green_candles_prev >= MIN_CONSECUTIVE_CANDLES) & (num_green_candles_prev <= MAX_PATTERN_LOOKBACK - 1))
            df_15min['trade_trigger_short'] = is_signal_candle_short.shift(1).fillna(False)
            df_15min['pattern_len_short'] = (num_green_candles_prev + 1).shift(1).fillna(0)
            
            # --- OPTIMIZATION: Populate the signal map ---
            long_triggers = df_15min.index[df_15min['trade_trigger_long']]
            for ts in long_triggers:
                signals_by_timestamp[ts].append({'symbol': symbol, 'direction': 'LONG'})
            short_triggers = df_15min.index[df_15min['trade_trigger_short']]
            for ts in short_triggers:
                signals_by_timestamp[ts].append({'symbol': symbol, 'direction': 'SHORT'})

            # Store patterns
            df_15min['pattern_high_long'] = df_15min['high'].rolling(MAX_PATTERN_LOOKBACK).max().shift(1)
            df_15min['pattern_low_long'] = df_15min['low'].rolling(MAX_PATTERN_LOOKBACK).min().shift(1)
            df_15min['pattern_high_short'] = df_15min['high'].rolling(MAX_PATTERN_LOOKBACK).max().shift(1)
            df_15min['pattern_low_short'] = df_15min['low'].rolling(MAX_PATTERN_LOOKBACK).min().shift(1)
            
            # Filter and add to map
            df_15min = df_15min[(df_15min.index >= START_DATE) & (df_15min.index < END_DATE)].copy()
            df_15min.dropna(inplace=True)
            all_data_map[symbol] = df_15min
        except Exception as e:
            logging.warning(f"Error loading or processing data for {symbol}: {e}")
            continue
    
    # --- RUN SIMULATION ---
    # The generation of all_timestamps is moved here to ensure it only includes
    # timestamps that exist in the loaded data.
    all_timestamps = sorted(list(set(ts for df in all_data_map.values() for ts in df.index)))
    
    logging.info(f"Starting simulation on {len(all_data_map)} symbols...")
    portfolio = PortfolioManager(INITIAL_CAPITAL, RISK_PER_TRADE_PCT, MAX_PORTFOLIO_RISK_PCT)
    trade_log = []
    rejected_log = []
    
    run_simulation(all_data_map, portfolio, trade_log, rejected_log, signals_by_timestamp)
    
    # --- FINAL LOGGING & SUMMARY ---
    final_trade_df = pd.DataFrame(trade_log)
    rejected_df = pd.DataFrame(rejected_log)

    trade_log_path = os.path.join(log_dir, 'trade_log.csv')
    final_trade_df.to_csv(trade_log_path, index=False)

    rejected_log_path = os.path.join(log_dir, 'rejected_trades.csv')
    rejected_df.to_csv(rejected_log_path, index=False)

    logging.info("\n--- Overall Backtest Summary ---")
    calculate_and_log_metrics(final_trade_df, "ALL_SYMBOLS_COMBINED", INITIAL_CAPITAL, log_dir)
    logging.info(f"Full trade log saved to: {trade_log_path}")
    logging.info(f"Rejected trades log saved to: {rejected_log_path}")
    logging.info("--- Backtest Simulation Finished ---")

if __name__ == "__main__":
    main()
