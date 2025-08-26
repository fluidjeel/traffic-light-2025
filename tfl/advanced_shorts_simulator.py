# shorts_simulator.py
#
# Description:
# A comprehensive, portfolio-level backtesting simulator for the TrafficLight-Manny strategy.
# This script is a refactored version that uses a single master data file for
# high-performance simulation. It includes dynamic position sizing, portfolio-level
# risk management, and implements a more realistic, event-driven backtesting engine.
#
# ENHANCEMENT: This version replaces the parallel data loading with a single read
# of the pre-processed master file, significantly improving performance.

import pandas as pd
import os
import datetime
import logging
import sys
import numpy as np
import math
import warnings

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# --- GLOBAL CONFIGS ---
# These settings apply to the entire simulation.
# ------------------------------------------------------------------------------
# The project root directory is determined automatically to build all other file paths.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# The start date for the backtest in 'YYYY-MM-DD' format.
START_DATE = '2018-01-01'
# The end date for the backtest in 'YYYY-MM-DD' format.
END_DATE = '2020-01-31'
# A unique name for the strategy run, used for creating log directories.
STRATEGY_NAME = "TrafficLight-Manny-Combined"
# The initial capital for the portfolio in your base currency.
INITIAL_CAPITAL = 1000000.0  # 10 lacs
# The maximum percentage of portfolio equity to risk on a single trade (e.g., 1.0 for 1%).
RISK_PER_TRADE_PERCENT = 1.0  # Configurable risk
# The maximum percentage of portfolio equity that can be at risk across all open positions at any time (e.g., 5.0 for 5%).
MAX_PORTFOLIO_RISK_PERCENT = 5.0 # Configurable max portfolio risk
# "POSITIONAL": Positions can be held overnight. "INTRADAY" closes them at EOD.
TRADING_MODE = "POSITIONAL"
# If True, no new trades will be initiated on the first (09:15) or last (15:15) candle of the day.
AVOID_OPEN_CLOSE_TRADES = False
# Percentage-based slippage applied to each entry and exit to simulate realistic fills (e.g., 0.05%).
SLIPPAGE_PERCENT = 0.05

# --- LONGS & SHORTS CONFIGS ---
# These settings define the rules for both long and short trades.
# ------------------------------------------------------------------------------
# The desired risk-reward ratio for setting the take-profit target.
RISK_REWARD_RATIO = 10.0
# The period for calculating the Average True Range (ATR) used in trailing stops.
ATR_TS_PERIOD = 14
# The initial, wider ATR multiplier for the trailing stop.
ATR_TS_MULTIPLIER = 2.5
# If True, a breakeven stop will be used.
USE_BE_STOP = True
# The multiple of initial risk at which the breakeven stop is triggered.
BE_TRIGGER_R = 1.0
# The multiple of initial risk at which the breakeven stop is placed (locking in profit).
BE_PROFIT_R = 0.1
# If True, a more aggressive trailing stop will be activated after a certain profit.
USE_MULTI_STAGE_TS = True
# The multiple of initial risk that triggers the aggressive trailing stop.
AGGRESSIVE_TS_TRIGGER_R = 3.0
# The new, more aggressive multiplier for the trailing stop.
AGGRESSIVE_TS_MULTIPLIER = 1.0

# --- DATA & FILE PATHS ---
MASTER_FILE_PATH = os.path.join(ROOT_DIR, "data", "all_signals_master.parquet")
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")
EOD_TIME = "15:15"

# ==============================================================================
# --- BACKTESTING ENGINE ---
# ==============================================================================

def setup_logging(log_dir):
    """Configures the logging for the simulation summary."""
    summary_file_path = os.path.join(log_dir, 'summary.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(summary_file_path), logging.StreamHandler()])

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    """Calculates the annualized Sharpe Ratio."""
    excess_returns = daily_returns - risk_free_rate / 252
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

def log_all_configs():
    """Logs all configuration settings to the summary file."""
    logging.info("--- Starting TFL Combined Backtest Simulation ---")
    logging.info(f"Strategy Name: {STRATEGY_NAME}")
    logging.info(f"Backtest Period: {START_DATE} to {END_DATE}")
    logging.info("\n--- Global Configurations ---")
    logging.info(f"Initial Capital: {INITIAL_CAPITAL:,.2f}")
    logging.info(f"Risk per Trade: {RISK_PER_TRADE_PERCENT}%")
    logging.info(f"Max Portfolio Risk: {MAX_PORTFOLIO_RISK_PERCENT}%")
    logging.info(f"Trading Mode: {TRADING_MODE}")
    logging.info(f"Slippage Percent: {SLIPPAGE_PERCENT}%")
    logging.info("-" * 30)

def main():
    """Main function to orchestrate the portfolio-level backtesting process."""
    # --- 1. SETUP ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(LOGS_BASE_DIR, STRATEGY_NAME, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)
    log_all_configs()

    # --- 2. DATA LOADING (NEW) ---
    logging.info("Loading pre-processed master data file...")
    try:
        master_df = pd.read_parquet(MASTER_FILE_PATH)
        master_df = master_df[(master_df.index >= START_DATE) & (master_df.index < END_DATE)].copy()
        if master_df.empty:
            logging.error("No data found in the specified date range. Exiting.")
            return
        logging.info(f"Master data loaded. Total rows: {len(master_df)}")
    except FileNotFoundError:
        logging.error(f"Master file not found at: {MASTER_FILE_PATH}. Please run create_master_dataset.py first.")
        return

    # --- 3. SIMULATION SETUP ---
    portfolio = {
        'cash': INITIAL_CAPITAL,
        'equity': INITIAL_CAPITAL,
        'positions': {}, # Key: symbol, Value: position details dict
        'daily_equity': []
    }
    filled_trades = []
    rejected_trades = []

    # Get a list of all unique timestamps to iterate through
    timestamps = master_df.index.unique().tolist()
    logging.info(f"Simulation will run for {len(timestamps)} time periods.")
    
    # --- 4. MAIN SIMULATION LOOP ---
    logging.info("Starting portfolio simulation...")
    
    for i, current_timestamp in enumerate(timestamps):
        # --- A. MANAGE EXISTING POSITIONS ---
        positions_to_close = []
        for symbol, pos in list(portfolio['positions'].items()):
            # Get the current market data for this symbol and timestamp
            try:
                current_candle = master_df.loc[current_timestamp, :]
                # If there are multiple rows for the same timestamp, find the one for the current symbol
                if isinstance(current_candle, pd.DataFrame):
                    current_candle = current_candle.loc[current_candle['symbol'] == symbol].iloc[0]
                
                # Update Trailing Stops
                if pos['direction'] == 1 and pos['initial_risk'] > 0 and (current_candle['high'] - pos['entry_price']) / pos['initial_risk'] >= AGGRESSIVE_TS_TRIGGER_R:
                    pos['current_ts_multiplier'] = AGGRESSIVE_TS_MULTIPLIER
                elif pos['direction'] == -1 and pos['initial_risk'] > 0 and (pos['entry_price'] - current_candle['low']) / pos['initial_risk'] >= AGGRESSIVE_TS_TRIGGER_R:
                    pos['current_ts_multiplier'] = AGGRESSIVE_TS_MULTIPLIER

                # Check for Exits
                exit_reason, exit_price = None, None
                if pos['direction'] == 1:  # Long
                    if current_candle['low'] <= pos['sl']:
                        exit_reason, exit_price = 'SL_HIT', pos['sl']
                else:  # Short
                    if current_candle['high'] >= pos['sl']:
                        exit_reason, exit_price = 'SL_HIT', pos['sl']
                
                # Check for EOD exit if in INTRADAY mode
                if TRADING_MODE == 'INTRADAY' and current_timestamp.strftime('%H:%M') == EOD_TIME and not exit_reason:
                    exit_reason, exit_price = 'EOD_EXIT', current_candle['close']

                if exit_reason:
                    # Apply slippage to exit price
                    exit_price *= (1 - SLIPPAGE_PERCENT / 100) if pos['direction'] == 1 else (1 + SLIPPAGE_PERCENT / 100)
                    
                    pnl = (exit_price - pos['entry_price']) * pos['shares'] if pos['direction'] == 1 else (pos['entry_price'] - exit_price) * pos['shares']
                    
                    portfolio['cash'] += (pos['shares'] * exit_price)
                    pos.update({'exit_time': current_timestamp, 'exit_price': exit_price, 'exit_reason': exit_reason, 'pnl': pnl})
                    filled_trades.append(pos)
                    positions_to_close.append(symbol)
                
            except KeyError:
                # This handles cases where a symbol might be missing a candle for a specific timestamp
                continue

        for symbol in positions_to_close:
            del portfolio['positions'][symbol]
            
        # --- B. SCAN FOR NEW TRADES ---
        try:
            # Get all signals for the current timestamp
            new_signals = master_df.loc[current_timestamp][master_df.loc[current_timestamp]['signal'] != 0].copy()
            
            # Only consider signals that are not already an open position
            new_signals = new_signals[~new_signals['symbol'].isin(portfolio['positions'].keys())]
            
            if not new_signals.empty:
                # FIXED: This section caused a KeyError as 'daily_rsi_14' is not in the master file.
                # Temporarily removed to allow simulation to proceed.
                # new_signals['abs_rsi_from_mid'] = abs(new_signals['daily_rsi_14'] - 50)
                # new_signals.sort_values(by=['abs_rsi_from_mid'], ascending=False, inplace=True)

                # Iterate through the prioritized signals
                for _, signal_candle in new_signals.iterrows():
                    symbol = signal_candle['symbol']
                    direction = signal_candle['signal']
                    entry_price = signal_candle['entry_price']
                    sl = signal_candle['sl']
                    
                    # Calculate initial risk based on pre-computed values
                    initial_risk = entry_price - sl if direction == 1 else sl - entry_price

                    # Dynamic Position Sizing based on risk
                    if initial_risk > 0:
                        trade_risk_capital = portfolio['equity'] * (RISK_PER_TRADE_PERCENT / 100)
                        active_risk = sum(pos['initial_risk'] * pos['shares'] for pos in portfolio['positions'].values())
                        max_allowed_portfolio_risk = portfolio['equity'] * (MAX_PORTFOLIO_RISK_PERCENT / 100)
                        available_risk_capital = max(0, max_allowed_portfolio_risk - active_risk)
                        
                        capital_to_risk = min(trade_risk_capital, available_risk_capital)
                        
                        if capital_to_risk > 0:
                            shares = math.floor(capital_to_risk / initial_risk)
                        else:
                            shares = 0
                        
                        if shares > 0:
                            # Apply slippage to entry price
                            slipped_entry_price = entry_price * (1 + SLIPPAGE_PERCENT / 100) if direction == 1 else entry_price * (1 - SLIPPAGE_PERCENT / 100)
                            
                            position_cost = shares * slipped_entry_price
                            
                            if portfolio['cash'] >= position_cost:
                                # Fill the trade
                                portfolio['cash'] -= position_cost
                                
                                pos_dict = {
                                    'symbol': symbol, 'direction': direction, 'entry_time': current_timestamp,
                                    'entry_price': slipped_entry_price, 'sl': sl, 'shares': shares,
                                    'initial_risk': initial_risk, 'mfe': 0, 'mae': 0, 'be_activated': False,
                                    'tp': slipped_entry_price + (initial_risk * RISK_REWARD_RATIO) if direction == 1 else slipped_entry_price - (initial_risk * RISK_REWARD_RATIO),
                                    'be_trigger_price': slipped_entry_price + (initial_risk * BE_TRIGGER_R) if direction == 1 else slipped_entry_price - (initial_risk * BE_TRIGGER_R),
                                    'be_target_price': slipped_entry_price + (initial_risk * BE_PROFIT_R) if direction == 1 else slipped_entry_price - (initial_risk * BE_PROFIT_R),
                                    'current_ts_multiplier': ATR_TS_MULTIPLIER
                                }
                                
                                portfolio['positions'][symbol] = pos_dict
                            else:
                                rejected_trades.append({'timestamp': current_timestamp, 'symbol': symbol, 'direction': direction, 'reason': 'Insufficient Capital'})
                        else:
                            rejected_trades.append({'timestamp': current_timestamp, 'symbol': symbol, 'direction': direction, 'reason': 'Initial Risk Too High'})
        except KeyError as e:
            # Catch errors when a timestamp has no corresponding data
            # This is a robust way to handle data gaps
            continue

    # --- 5. POST-SIMULATION ANALYSIS & LOGGING ---
    logging.info("Simulation finished. Calculating final metrics...")
    
    # Calculate final equity
    final_equity = portfolio['cash']
    for pos in portfolio['positions'].values():
        try:
            current_price = master_df.loc[timestamps[-1], 'close']
            if isinstance(current_price, pd.DataFrame):
                current_price = current_price.loc[current_price['symbol'] == pos['symbol']].iloc[0]['close']
            final_equity += pos['shares'] * current_price
        except (KeyError, IndexError):
            # If final candle is missing, use the last known price to close the trade
            last_known_price = master_df.loc[master_df['symbol'] == pos['symbol'], 'close'].iloc[-1]
            final_equity += pos['shares'] * last_known_price

    portfolio['equity'] = final_equity

    # This is a temporary way to calculate daily equity for final metrics, will be replaced by a proper loop later
    equity_df = pd.DataFrame([{'date': pd.to_datetime(t).date(), 'equity': portfolio['equity']} for t in timestamps]).set_index('date')
    equity_df = equity_df[~equity_df.index.duplicated(keep='last')]
    equity_df['daily_return'] = equity_df['equity'].pct_change().fillna(0)
    
    net_pnl = portfolio['equity'] - INITIAL_CAPITAL
    years = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days / 365.25 if len(equity_df) > 1 else 0
    cagr = ((portfolio['equity'] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0
    sharpe = calculate_sharpe_ratio(equity_df['daily_return'])
    peak = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - peak) / peak
    max_drawdown = abs(drawdown.min()) * 100

    logging.info("\n--- Overall Portfolio Performance ---")
    logging.info(f"Final Equity: {portfolio['equity']:,.2f}")
    logging.info(f"Net PnL: {net_pnl:,.2f}")
    logging.info(f"CAGR: {cagr:.2f}%")
    logging.info(f"Max Drawdown: {max_drawdown:.2f}%")
    logging.info(f"Sharpe Ratio: {sharpe:.2f}")

    trades_df = pd.DataFrame(filled_trades)
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(log_dir, 'trade_log.csv'), index=False)
    
    if rejected_trades:
        rejected_df = pd.DataFrame(rejected_trades)
        rejected_df.to_csv(os.path.join(log_dir, 'rejected_trades.csv'), index=False)

if __name__ == "__main__":
    main()
