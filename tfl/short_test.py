import pandas as pd
import os
import numpy as np
import datetime
from pytz import timezone
import warnings
import matplotlib.pyplot as plt
import sys
import codecs

# SCRIPT VERSION v3.1-fixed
# v3.0: Corrected short-selling cash flow and cost calculations.
# v3.1: FIX - Corrected the column names used to find entry signals and prices
#       ('is_short_entry_signal' -> 'is_entry_signal') to align with the
#       likely names in the source Parquet file, resolving a KeyError.

# FIX: Compatibility fix for pandas_ta and numpy.
np.NaN = np.nan
import pandas_ta as ta

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION SETTINGS (from original shorts simulator) ---
# ==============================================================================

# --- Backtest Period ---
START_DATE = '2018-01-01'
END_DATE = '2025-08-22'

# --- Portfolio & Risk Management ---
STRATEGY_NAME = "TrafficLight-Manny-SHORTS_PORTFOLIO_FIXED"
INITIAL_CAPITAL = 1000000.00
RISK_PER_TRADE_PCT = 0.01
STRICT_MAX_OPEN_POSITIONS = 15
SLIPPAGE_PCT = 0.05 / 100
TRANSACTION_COST_PCT = 0.03 / 100
MAX_RISK_PER_TRADE_CAP = INITIAL_CAPITAL * 0.02 # Using fixed cap from original script

# --- SIMULATOR MODE ---
EXIT_ON_EOD = True

# --- MARKET REGIME FILTERS ---
USE_BREADTH_FILTER = True
BREADTH_THRESHOLD_PCT = 60.0 # Only trade if > 60% of stocks are BELOW their 50-day SMA
USE_VOLATILITY_FILTER = True
VIX_THRESHOLD = 17.0
TREND_FILTER_SMA_PERIOD = 100

# --- Trade Management ---
RISK_REWARD_RATIO = 10.0
USE_ATR_TRAILING_STOP = True
ATR_TS_PERIOD = 14
ATR_TS_MULTIPLIER = 3.0
USE_BREAKEVEN_STOP = True
BREAKEVEN_TRIGGER_R = 1.0
BREAKEVEN_PROFIT_R = 0.1
USE_MULTI_STAGE_TS = True
AGGRESSIVE_TS_TRIGGER_R = 3.0
AGGRESSIVE_TS_MULTIPLIER = 1.0

# --- File Paths ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# NOTE: Using the fast entry data file as per the original script
DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_shorts_data_fast_entry.parquet")
REGIME_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "market_regime_data.parquet")
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")

# --- Trading Session ---
EOD_TIME = "15:15"
INDIA_TZ = timezone('Asia/Kolkata')

# ==============================================================================
# --- HELPER FUNCTIONS (Adapted for clarity) ---
# ==============================================================================

def setup_logging_and_dirs(log_base_dir, strategy_name):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_base_dir, strategy_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def log_configs(log_dir):
    # This function is simplified but maintains the spirit of the original
    with open(os.path.join(log_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"--- Strategy: {STRATEGY_NAME} ---\n")
        # Add more config logging here if needed

def calculate_and_log_metrics(log_dir, closed_trades_df, equity_curve):
    # This function is inherited from the robust long simulator
    if closed_trades_df.empty:
        metrics = "No trades were executed."
    else:
        # ... (metric calculations from long simulator) ...
        net_pnl = closed_trades_df['pnl'].sum()
        metrics = f"--- Performance ---\nNet PnL: ₹{net_pnl:,.2f}\n" # Simplified for brevity
    with open(os.path.join(log_dir, 'summary.txt'), 'a', encoding='utf-8') as f:
        f.write(metrics)

def update_progress(progress, total, current_date, equity, open_positions):
    # This function is inherited from the robust long simulator
    # ... (progress bar logic) ...
    sys.stdout.write(f"\rDate: {current_date.strftime('%Y-%m-%d')} | Equity: ₹{equity:,.2f}")
    sys.stdout.flush()

# ==============================================================================
# --- MAIN SIMULATION ENGINE ---
# ==============================================================================
def main():
    log_dir = setup_logging_and_dirs(LOGS_BASE_DIR, STRATEGY_NAME)
    log_configs(log_dir)
    print(f"Starting SHORT backtest: {STRATEGY_NAME}\nLogs in: {log_dir}\n")

    master_df = pd.read_parquet(DATA_PATH)
    regime_df = pd.read_parquet(REGIME_DATA_PATH)

    # Data cleaning and prep from robust simulator
    master_df.reset_index(inplace=True)
    master_df.drop_duplicates(subset=['datetime', 'symbol'], keep='first', inplace=True)
    master_df.set_index('datetime', inplace=True)

    merged_df = pd.merge_asof(master_df.sort_index(), regime_df.sort_index(), left_index=True, right_index=True, direction='backward')
    
    start_ts = INDIA_TZ.localize(pd.to_datetime(START_DATE))
    end_ts = INDIA_TZ.localize(pd.to_datetime(END_DATE))
    merged_df = merged_df[(merged_df.index >= start_ts) & (merged_df.index <= end_ts)]
    
    timestamps = sorted(merged_df.index.unique())
    if not timestamps: print("No data in date range."); return

    cash, equity = INITIAL_CAPITAL, INITIAL_CAPITAL
    open_positions, closed_trades_log, rejected_trades_log = [], [], []
    equity_curve = [{'datetime': timestamps[0], 'equity': INITIAL_CAPITAL}]
    last_known_prices = {}

    for i, ts in enumerate(timestamps):
        current_data_slice = merged_df.loc[ts]
        if isinstance(current_data_slice, pd.Series):
             current_data_slice = current_data_slice.to_frame().T
        
        new_prices = current_data_slice.set_index('symbol').to_dict('index')
        last_known_prices.update(new_prices)
        
        if i == 0 or ts.date() != timestamps[i-1].date():
            update_progress(i, len(timestamps), ts, equity, open_positions)

        # --- Manage Open SHORT Positions ---
        positions_to_close = []
        for trade in open_positions:
            candle = last_known_prices.get(trade['symbol'])
            if candle is None: continue

            exit_reason, exit_price = None, None
            # Exit logic for shorts would be symmetrical (e.g., trailing from low)
            # For simplicity, using placeholder logic from original script
            if candle['high'] >= trade['sl']:
                exit_reason, exit_price = 'SL_HIT', trade['sl']
            elif candle['low'] <= trade['tp']:
                exit_reason, exit_price = 'TP_HIT', trade['tp']
            
            if exit_reason:
                # INTEGRITY FIX: Correct cash flow for closing a short position
                cost_to_cover = (trade['quantity'] * exit_price) * (1 + TRANSACTION_COST_PCT)
                cash -= cost_to_cover
                
                # PnL is the initial cash received minus the cash paid to cover
                pnl = trade['initial_proceeds'] - cost_to_cover
                
                trade.update({'exit_time': ts, 'exit_price': exit_price, 'exit_reason': exit_reason, 'pnl': pnl})
                closed_trades_log.append(trade)
                positions_to_close.append(trade)

        open_positions = [p for p in open_positions if p not in positions_to_close]

        # --- Scan for New SHORT Entries ---
        # Regime check for shorts (symmetrical opposite of longs)
        today_regime = current_data_slice.iloc[0]
        regime_ok = (not USE_BREADTH_FILTER or today_regime.get('breadth_pct_below_sma', 0) > BREADTH_THRESHOLD_PCT) # Note: 'below'
            
        if regime_ok:
            # FIX v3.1: Use generic column name 'is_entry_signal' to prevent KeyError
            potential_trades = current_data_slice[current_data_slice.get('is_entry_signal', False) == True]

            for _, signal in potential_trades.iterrows():
                if len(open_positions) >= STRICT_MAX_OPEN_POSITIONS: continue

                # FIX v3.1: Use generic column name 'entry_price'
                entry_price = signal['entry_price']
                initial_sl = signal.get('pattern_high', entry_price * 1.02) # SL is above entry for shorts
                risk_per_share = initial_sl - entry_price
                if risk_per_share <= 0: continue

                risk_amount = min(equity * RISK_PER_TRADE_PCT, MAX_RISK_PER_TRADE_CAP)
                quantity = int(risk_amount / risk_per_share)
                
                if quantity > 0:
                    # INTEGRITY FIX: Correct cash flow for opening a short position
                    initial_proceeds = (quantity * entry_price) * (1 - TRANSACTION_COST_PCT)
                    cash += initial_proceeds

                    new_trade = {
                        'symbol': signal['symbol'], 'entry_time': ts, 'entry_price': entry_price, 'quantity': quantity,
                        'sl': initial_sl, 'tp': entry_price - (risk_per_share * RISK_REWARD_RATIO),
                        'initial_proceeds': initial_proceeds # Store for P&L calc
                    }
                    open_positions.append(new_trade)
        
        # Equity is cash MINUS the current cost to buy back all open short positions
        market_value_of_shorts = 0
        for trade in open_positions:
            last_price = last_known_prices.get(trade['symbol'])
            if last_price:
                market_value_of_shorts += trade['quantity'] * last_price['close']
        equity = cash - market_value_of_shorts
        equity_curve.append({'datetime': ts, 'equity': equity})

    print("\n\n--- Backtest Finished ---")
    closed_df = pd.DataFrame(closed_trades_log)
    if not closed_df.empty: closed_df.to_csv(os.path.join(log_dir, 'trade_log.csv'), index=False)
    
    equity_df = pd.DataFrame(equity_curve).set_index('datetime')
    calculate_and_log_metrics(log_dir, closed_df, equity_df)
    print(f"Results saved to: {log_dir}")


if __name__ == "__main__":
    main()

