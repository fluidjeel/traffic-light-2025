import pandas as pd
import os
import numpy as np
import datetime
from pytz import timezone
import warnings
import matplotlib.pyplot as plt
import sys
import codecs # Import codecs for the fix

# SCRIPT VERSION v3.7
#
# CRITICAL BUG FIX (In response to user error report):
# v3.7: - Fixed a "ValueError: cannot convert float NaN to integer" that occurred
#         during position sizing.
#       - The bug was caused by the simulator failing to find the 'pattern_low'
#         (for the stop-loss) on the entry candle's data row.
#       - The fix implements a data preparation step after loading, which creates
#         a new 'pattern_low_for_sl' column that correctly carries forward the
#         stop-loss level from the setup candle to the entry candle.
#
# PREVIOUS BUG FIX:
# v3.6: - Fixed a TypeError in the final metrics calculation.
# v3.5: - Corrected unrealistic entry prices on market open gaps.
# v3.4: - Corrected P&L calculation and added slippage to long entries.

# Fix for pandas_ta compatibility
np.NaN = np.nan
import pandas_ta as ta

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# --- Backtest Period ---
START_DATE = '2018-01-01'
END_DATE = '2025-08-22'

# --- Portfolio & Risk Management ---
STRATEGY_NAME = "TrafficLight-Manny-LONGS_PORTFOLIO_v3.7"
ENTRY_TYPE_TO_USE = 'fast'
INITIAL_CAPITAL = 1000000.00
RISK_PER_TRADE_PCT = 0.01
MAX_TOTAL_RISK_PCT = 0.05
MAX_CAPITAL_PER_TRADE_PCT = 0.25
STRICT_MAX_OPEN_POSITIONS = 10
SLIPPAGE_PCT = 0.05 / 100
TRANSACTION_COST_PCT = 0.03 / 100

# --- SIMULATOR MODE ---
EXIT_ON_EOD = True
ALLOW_AFTERNOON_POSITIONAL = False
AFTERNOON_ENTRY_THRESHOLD = "13:00"
AVOID_OPEN_CLOSE_ENTRIES = True

# --- MARKET REGIME FILTERS ---
USE_BREADTH_FILTER = True
BREADTH_THRESHOLD_PCT = 60.0
USE_VOLATILITY_FILTER = False
VIX_THRESHOLD = 17.0
TREND_FILTER_SMA_PERIOD = 30

# --- Trade Management ---
RISK_REWARD_RATIO = 10.0
USE_ATR_TRAILING_STOP = True
ATR_TS_PERIOD = 14
ATR_TS_MULTIPLIER = 4.0
USE_BREAKEVEN_STOP = True
BREAKEVEN_TRIGGER_R = 1.0
BREAKEVEN_PROFIT_R = 0.1
USE_MULTI_STAGE_TS = True
AGGRESSIVE_TS_TRIGGER_R = 5.0
AGGRESSIVE_TS_MULTIPLIER = 1.0

# --- File Paths (Script must be in 'tfl' folder) ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_longs_data_with_signals_and_atr.parquet")
REGIME_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "market_regime_data.parquet")
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")

# --- Trading Session ---
EOD_TIME = "15:15"
INDIA_TZ = timezone('Asia/Kolkata')

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

def setup_logging_and_dirs(log_base_dir, strategy_name):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_base_dir, strategy_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def log_configs(log_dir):
    config_vars = {
        'STRATEGY_NAME': STRATEGY_NAME, 'BACKTEST_PERIOD': f"{START_DATE} to {END_DATE}",
        'ENTRY_TYPE_TO_USE': ENTRY_TYPE_TO_USE, 'INITIAL_CAPITAL': f"₹{INITIAL_CAPITAL:,.2f}",
        'RISK_PER_TRADE_PCT': f"{RISK_PER_TRADE_PCT*100:.2f}%", 'MAX_TOTAL_RISK_PCT': f"{MAX_TOTAL_RISK_PCT*100:.2f}%",
        'MAX_CAPITAL_PER_TRADE_PCT': f"{MAX_CAPITAL_PER_TRADE_PCT*100:.2f}%",
        'STRICT_MAX_OPEN_POSITIONS': STRICT_MAX_OPEN_POSITIONS, 'EXIT_ON_EOD': EXIT_ON_EOD,
        'ALLOW_AFTERNOON_POSITIONAL': f"{ALLOW_AFTERNOON_POSITIONAL} (After {AFTERNOON_ENTRY_THRESHOLD})",
        'AVOID_OPEN_CLOSE_ENTRIES': AVOID_OPEN_CLOSE_ENTRIES
    }
    market_regime_vars = {
        'USE_BREADTH_FILTER': f"{USE_BREADTH_FILTER} (>{BREADTH_THRESHOLD_PCT}%)",
        'USE_VOLATILITY_FILTER': f"{USE_VOLATILITY_FILTER} (VIX > {VIX_THRESHOLD})",
        'TREND_FILTER_SMA_PERIOD': f"Nifty > SMA({TREND_FILTER_SMA_PERIOD})" if TREND_FILTER_SMA_PERIOD > 0 else "Disabled",
    }
    with open(os.path.join(log_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("--- Configuration Settings ---\n")
        for key, value in config_vars.items(): f.write(f"- {key}: {value}\n")
        f.write("\n--- Market Regime Filters ---\n")
        for key, value in market_regime_vars.items(): f.write(f"- {key}: {value}\n")
        f.write("-" * 30 + "\n\n")

def calculate_and_log_metrics(log_dir, closed_trades_df, equity_curve_df):
    if closed_trades_df.empty:
        metrics = "No trades were executed."
    else:
        total_trades = len(closed_trades_df)
        wins = closed_trades_df[closed_trades_df['pnl'] > 0]
        win_rate = (len(wins) / total_trades) * 100
        gross_profit = wins['pnl'].sum()
        gross_loss = closed_trades_df[closed_trades_df['pnl'] <= 0]['pnl'].sum()
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        net_pnl = closed_trades_df['pnl'].sum()
        
        equity_curve_df['peak'] = equity_curve_df['equity'].cummax()
        equity_curve_df['drawdown'] = (equity_curve_df['equity'] - equity_curve_df['peak']) / equity_curve_df['peak']
        max_drawdown = equity_curve_df['drawdown'].min() * 100
        
        metrics = (
            f"--- Performance Metrics ---\n"
            f"- Total Trades: {total_trades}\n- Win Rate: {win_rate:.2f}%\n"
            f"- Profit Factor: {profit_factor:.2f}\n- Net PnL: ₹{net_pnl:,.2f}\n"
            f"- Max Drawdown: {max_drawdown:.2f}%\n"
        )
    with open(os.path.join(log_dir, 'summary.txt'), 'a', encoding='utf-8') as f:
        f.write(metrics)

def update_progress(progress, total, current_date, equity, open_positions):
    bar_length, percent = 40, float(progress) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    equity_str, open_pos_str = f"Equity: ₹{equity:,.2f}", f"Open Positions: {len(open_positions)}"
    sys.stdout.write(f"\rDate: {current_date.strftime('%Y-%m-%d')} | {equity_str} | {open_pos_str} | Progress: [{arrow + spaces}] {percent*100:.1f}%")
    sys.stdout.flush()

# ==============================================================================
# --- MAIN SIMULATION ENGINE ---
# ==============================================================================

def main():
    log_dir = setup_logging_and_dirs(LOGS_BASE_DIR, STRATEGY_NAME)
    log_configs(log_dir)
    print(f"Starting backtest: {STRATEGY_NAME}. Log files will be saved in:\n{log_dir}\n")

    try:
        master_df = pd.read_parquet(DATA_PATH)
        regime_df = pd.read_parquet(REGIME_DATA_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: Data file not found. Details: {e}"); return

    master_df.reset_index(inplace=True)
    master_df.drop_duplicates(subset=['datetime', 'symbol'], keep='first', inplace=True)
    master_df.set_index('datetime', inplace=True)

    # BUG FIX v3.7: Prepare the data after loading to create a dedicated column
    # for the entry's stop-loss. This carries the 'pattern_low' from the setup
    # candle (T-1) forward to the entry candle (T).
    master_df.sort_values(by=['symbol', 'datetime'], inplace=True)
    master_df['pattern_low_for_sl'] = master_df.groupby('symbol')['pattern_low'].shift(1)

    merged_df = pd.merge_asof(master_df.sort_index(), regime_df.sort_index(), left_index=True, right_index=True, direction='backward')
    
    start_ts = INDIA_TZ.localize(pd.to_datetime(START_DATE))
    end_ts = INDIA_TZ.localize(pd.to_datetime(END_DATE))
    merged_df = merged_df[(merged_df.index >= start_ts) & (merged_df.index <= end_ts)]
    
    timestamps = sorted(merged_df.index.unique())
    if not timestamps: print("No data available for the specified date range. Exiting."); return
    
    cash, equity = INITIAL_CAPITAL, INITIAL_CAPITAL
    open_positions, closed_trades_log, rejected_trades_log = [], [], []
    equity_curve = [{'datetime': timestamps[0], 'equity': INITIAL_CAPITAL}]
    
    last_known_prices = {}
    
    total_timestamps = len(timestamps)
    for i, ts in enumerate(timestamps):
        current_data_slice = merged_df.loc[ts]
        if isinstance(current_data_slice, pd.Series):
             current_data_slice = current_data_slice.to_frame().T
        
        new_prices = current_data_slice.set_index('symbol').to_dict('index')
        last_known_prices.update(new_prices)
        
        if i == 0 or ts.date() != timestamps[i-1].date():
            update_progress(i + 1, total_timestamps, ts, equity, open_positions)
            
        positions_to_close = []
        for trade in open_positions:
            candle = last_known_prices.get(trade['symbol'])
            if candle is None: continue
            
            exit_reason, exit_price = None, None
            current_profit_r = (candle['high'] - trade['entry_price']) / trade['initial_risk_per_share'] if trade['initial_risk_per_share'] > 0 else 0
            
            if USE_BREAKEVEN_STOP and not trade['be_activated'] and current_profit_r >= BREAKEVEN_TRIGGER_R:
                trade['sl'] = trade['entry_price'] + (trade['initial_risk_per_share'] * BREAKEVEN_PROFIT_R); trade['be_activated'] = True
            if USE_MULTI_STAGE_TS and current_profit_r >= AGGRESSIVE_TS_TRIGGER_R:
                trade['current_ts_multiplier'] = AGGRESSIVE_TS_MULTIPLIER
            if USE_ATR_TRAILING_STOP and pd.notna(candle[f'atr_{ATR_TS_PERIOD}']):
                trade['sl'] = max(trade['sl'], candle['high'] - (candle[f'atr_{ATR_TS_PERIOD}'] * trade['current_ts_multiplier']))

            if candle['low'] <= trade['sl']:
                exit_reason, exit_price = 'TS_HIT' if trade['sl'] > trade['initial_sl'] else 'SL_HIT', trade['sl']
            elif candle['high'] >= trade['tp']:
                exit_reason, exit_price = 'TP_HIT', trade['tp']
            
            elif EXIT_ON_EOD and ts.strftime('%H:%M') == EOD_TIME:
                 if ALLOW_AFTERNOON_POSITIONAL and trade.get('is_afternoon_entry', False):
                     continue 
                 exit_reason, exit_price = 'EOD_EXIT', candle['close']
            
            if exit_reason:
                gross_proceeds = trade['quantity'] * exit_price
                net_proceeds = gross_proceeds * (1 - TRANSACTION_COST_PCT)
                cash += net_proceeds
                
                initial_cost = trade['initial_cost_with_fees']
                pnl = net_proceeds - initial_cost

                trade.update({'exit_time': ts, 'exit_price': exit_price, 'exit_reason': exit_reason, 'pnl': pnl})
                closed_trades_log.append(trade)
                positions_to_close.append(trade)

        open_positions = [p for p in open_positions if p not in positions_to_close]
        total_current_risk_value = sum(trade.get('initial_risk_value', 0) for trade in open_positions)

        today_regime = current_data_slice.iloc[0]
        breadth_ok = not USE_BREADTH_FILTER or today_regime.get('breadth_pct_above_sma', 0) > BREADTH_THRESHOLD_PCT
        volatility_ok = not USE_VOLATILITY_FILTER or today_regime.get('india_vix_close', 0) > VIX_THRESHOLD
        trend_ok = TREND_FILTER_SMA_PERIOD <= 0 or today_regime.get(f'is_nifty_above_sma_{TREND_FILTER_SMA_PERIOD}', False)
        regime_ok = breadth_ok and volatility_ok and trend_ok
            
        if regime_ok:
            current_time_str = ts.strftime('%H:%M')
            if AVOID_OPEN_CLOSE_ENTRIES and (current_time_str == '09:15' or current_time_str == '15:15'):
                pass
            else:
                equity_for_risk_calc = equity
                entry_signal_col = f'is_{ENTRY_TYPE_TO_USE}_entry'
                price_col = f'{ENTRY_TYPE_TO_USE}_entry_price'
                potential_trades = current_data_slice[current_data_slice[entry_signal_col] == True].sort_values(by='daily_rsi', ascending=False)
                
                active_symbols = [p['symbol'] for p in open_positions]

                for _, signal in potential_trades.iterrows():
                    if signal['symbol'] in active_symbols:
                        continue
                        
                    if len(open_positions) >= STRICT_MAX_OPEN_POSITIONS:
                        rejected_trades_log.append({'timestamp': ts, 'symbol': signal['symbol'], 'reason': 'MAX_POSITIONS_REACHED'}); continue
                    
                    available_risk_budget = (equity_for_risk_calc * MAX_TOTAL_RISK_PCT) - total_current_risk_value
                    desired_risk_amount = equity_for_risk_calc * RISK_PER_TRADE_PCT
                    risk_amount = min(desired_risk_amount, available_risk_budget)

                    if risk_amount <= 0:
                        rejected_trades_log.append({'timestamp': ts, 'symbol': signal['symbol'], 'reason': 'MAX_TOTAL_RISK_EXCEEDED'}); break
                    
                    base_entry_price = signal[price_col]
                    realistic_base_entry_price = max(base_entry_price, signal['open'])
                    entry_price = realistic_base_entry_price * (1 + SLIPPAGE_PCT)
                    
                    # BUG FIX v3.7: Use the new 'pattern_low_for_sl' column to get the
                    # correct stop-loss level from the setup candle.
                    initial_sl = signal.get('pattern_low_for_sl', entry_price * 0.98)
                    
                    # Add a final check to ensure SL is a valid number before proceeding
                    if pd.isna(initial_sl):
                        rejected_trades_log.append({'timestamp': ts, 'symbol': signal['symbol'], 'reason': 'MISSING_SL_VALUE'})
                        continue
                        
                    risk_per_share = entry_price - initial_sl
                    if risk_per_share <= 0: continue
                    
                    quantity_by_risk = int(risk_amount / risk_per_share)
                    capital_for_trade = equity_for_risk_calc * MAX_CAPITAL_PER_TRADE_PCT
                    quantity_by_capital = int(capital_for_trade / entry_price) if entry_price > 0 else 0
                    quantity = min(quantity_by_risk, quantity_by_capital)
                    
                    cost = quantity * entry_price * (1 + TRANSACTION_COST_PCT)
                    
                    if quantity > 0 and cash >= cost:
                        cash -= cost
                        actual_risk_value = quantity * risk_per_share
                        
                        is_afternoon_entry = ts.strftime('%H:%M') >= AFTERNOON_ENTRY_THRESHOLD

                        new_trade = {
                            'symbol': signal['symbol'], 'entry_time': ts, 'entry_price': entry_price, 'quantity': quantity,
                            'sl': initial_sl, 'tp': entry_price + (risk_per_share * RISK_REWARD_RATIO),
                            'initial_risk_per_share': risk_per_share, 'initial_risk_value': actual_risk_value,
                            'initial_sl': initial_sl, 'be_activated': False, 'current_ts_multiplier': ATR_TS_MULTIPLIER,
                            'is_afternoon_entry': is_afternoon_entry,
                            'initial_cost_with_fees': cost
                        }
                        open_positions.append(new_trade)
                        total_current_risk_value += actual_risk_value
                        active_symbols.append(signal['symbol'])
                    else:
                        reason = 'INSUFFICIENT_CAPITAL'
                        if quantity_by_risk > quantity_by_capital and cash < (quantity_by_capital * entry_price):
                             reason = 'INSUFFICIENT_CAPITAL_FOR_CAPPED_SIZE'
                        rejected_trades_log.append({'timestamp': ts, 'symbol': signal['symbol'], 'reason': reason})
        
        market_value = 0
        for trade in open_positions:
            last_price_info = last_known_prices.get(trade['symbol'])
            if last_price_info is not None:
                market_value += trade['quantity'] * last_price_info['close']
        equity = cash + market_value
        equity_curve.append({'datetime': ts, 'equity': equity})
        
    update_progress(total_timestamps, total_timestamps, timestamps[-1], equity, open_positions)
    print("\n\n--- Backtest Simulation Finished ---")

    closed_df = pd.DataFrame(closed_trades_log)
    rejected_df = pd.DataFrame(rejected_trades_log)
    
    equity_df = pd.DataFrame(equity_curve).set_index('datetime')
    
    if not closed_df.empty: closed_df.to_csv(os.path.join(log_dir, 'trade_log.csv'), index=False)
    if not rejected_df.empty: rejected_df.to_csv(os.path.join(log_dir, 'rejected_trades_log.csv'), index=False)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    equity_df['equity'].plot(figsize=(12, 8), title='Portfolio Equity Curve', color='blue')
    plt.ylabel('Equity (₹)'); plt.xlabel('Date'); plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'equity_curve.png'))
    
    calculate_and_log_metrics(log_dir, closed_df, equity_df)
    print(f"Results and logs have been saved to: {log_dir}")

if __name__ == "__main__":
    main()

