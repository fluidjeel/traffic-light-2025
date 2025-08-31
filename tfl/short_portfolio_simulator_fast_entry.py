# short_portfolio_simulator_fast_entry.py (CORRECTLY ENHANCED v4)
#
# Description:
# An enhanced version of the shorts simulator that incorporates advanced entry
# and exit timing rules ported from the v3.3 long simulator.
#
# ENHANCEMENTS (Additive Changes):
# v4: - CRITICAL BUG FIX: Completely overhauled the cash and P&L logic to
#       correctly model the cash flow of a short-selling strategy. The simulator
#       now correctly ADDS cash on entry and SUBTRACTS cash on exit, resolving
#       the catastrophic portfolio failure.
# v3: - Added a dynamic progress bar for user feedback.
# v2: - Added `ALLOW_AFTERNOON_POSITIONAL` and `AVOID_OPEN_CLOSE_ENTRIES` flags.
#
# BUG FIX:
# - Corrected a KeyError by changing the entry price column to 'alert_candle_low'.

import os
import pandas as pd
import numpy as np
import datetime
from pytz import timezone
import sys
import matplotlib.pyplot as plt

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# --- Backtest Period ---
START_DATE = '2018-01-01'
END_DATE = '2025-08-22'

# --- Portfolio & Risk Management ---
STRATEGY_NAME = "TrafficLight-Manny-SHORTS_PORTFOLIO_CORRECTLY_ENHANCED"
INITIAL_CAPITAL = 1000000.00
RISK_PER_TRADE_PCT = 0.01
STRICT_MAX_OPEN_POSITIONS = 15
SLIPPAGE_PCT = 0.05
TRANSACTION_COST_PCT = 0.03 # Models brokerage, STT, fees, etc.
# Re-introduced to prevent runaway compounding
MAX_RISK_PER_TRADE_CAP = INITIAL_CAPITAL * 0.2

# --- SIMULATOR MODE ---
EXIT_ON_EOD = True
# NEW: If True, trades entered after the threshold time will not be force-closed at EOD.
ALLOW_AFTERNOON_POSITIONAL = False
AFTERNOON_ENTRY_THRESHOLD = "14:00"
# NEW: If True, prevents taking new entries on the first (9:15) and last (15:15) candles.
AVOID_OPEN_CLOSE_ENTRIES = True


# --- MARKET REGIME FILTERS ---
USE_BREADTH_FILTER = True
BREADTH_THRESHOLD_PCT = 60.0 # Only trade if > 60% of stocks are below their 50-day SMA
USE_VOLATILITY_FILTER = False
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
DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_shorts_data_fast_entry.parquet")
REGIME_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "market_regime_data.parquet")
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")

# --- Trading Session ---
EOD_TIME = "15:15"
INDIA_TZ = timezone('Asia/Kolkata')


# ==============================================================================
# --- HELPER FUNCTIONS (Preserved from original script) ---
# ==============================================================================

def setup_logging_and_dirs(log_base_dir, strategy_name):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_base_dir, strategy_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def log_summary(log_dir, closed_trades_df, equity_curve):
    if closed_trades_df.empty:
        summary = "No trades were executed during the backtest period."
    else:
        total_trades = len(closed_trades_df)
        wins = closed_trades_df[closed_trades_df['pnl'] > 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        gross_profit = wins['pnl'].sum()
        gross_loss = closed_trades_df[closed_trades_df['pnl'] <= 0]['pnl'].sum()
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        net_pnl = closed_trades_df['pnl'].sum()

        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['peak']) / equity_curve['peak']
        max_drawdown = equity_curve['drawdown'].min()
        
        summary = f"""
        --- Backtest Summary: {STRATEGY_NAME} ---
        Period: {START_DATE} to {END_DATE}
        --- Performance Metrics ---
        Net PnL:                  {net_pnl:,.2f}
        Total Trades:             {total_trades}
        Win Rate:                 {win_rate:.2f}%
        Profit Factor:            {profit_factor:.2f}
        Max Drawdown:             {max_drawdown:.2%}
    """
    with open(os.path.join(log_dir, 'summary.txt'), 'a') as f:
        f.write(summary)


def log_configs(log_dir):
    config_str = f"""
    --- Initial Configuration ---
    Strategy Name: {STRATEGY_NAME}
    Start Date: {START_DATE}
    End Date: {END_DATE}
    Initial Capital: {INITIAL_CAPITAL}
    Risk Per Trade: {RISK_PER_TRADE_PCT}
    Strict Max Positions: {STRICT_MAX_OPEN_POSITIONS}
    Slippage: {SLIPPAGE_PCT}
    Transaction Costs: {TRANSACTION_COST_PCT}
    Trading Mode: {'Intraday' if EXIT_ON_EOD else 'Positional'}
    Max Risk Per Trade Cap: {MAX_RISK_PER_TRADE_CAP}
    ALLOW_AFTERNOON_POSITIONAL: {ALLOW_AFTERNOON_POSITIONAL} (After {AFTERNOON_ENTRY_THRESHOLD})
    AVOID_OPEN_CLOSE_ENTRIES: {AVOID_OPEN_CLOSE_ENTRIES}
    --- Regime Filters ---
    USE_BREADTH_FILTER: {USE_BREADTH_FILTER} (Threshold: > {BREADTH_THRESHOLD_PCT}%)
    USE_VOLATILITY_FILTER: {USE_VOLATILITY_FILTER} (Threshold: > {VIX_THRESHOLD})
    TREND_FILTER_SMA_PERIOD: {TREND_FILTER_SMA_PERIOD}
    --- Trade Management ---
    Risk/Reward Ratio: {RISK_REWARD_RATIO}
    ATR Trailing Stop: {USE_ATR_TRAILING_STOP} (Period: {ATR_TS_PERIOD}, Multiplier: {ATR_TS_MULTIPLIER})
    Breakeven Stop: {USE_BREAKEVEN_STOP} (Trigger: {BREAKEVEN_TRIGGER_R}R, Profit: {BREAKEVEN_PROFIT_R}R)
    Multi-Stage TS: {USE_MULTI_STAGE_TS} (Trigger: {AGGRESSIVE_TS_TRIGGER_R}R, Multiplier: {AGGRESSIVE_TS_MULTIPLIER})
    """
    with open(os.path.join(log_dir, 'summary.txt'), 'w') as f:
        f.write(config_str)

# NEW HELPER FUNCTION (Ported from Longs Simulator)
def update_progress(progress, total, current_date, equity, open_positions):
    """Displays a dynamic progress bar in the console."""
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
    print(f"Starting SHORT backtest: {STRATEGY_NAME}\nLogs in: {log_dir}\n")

    master_df = pd.read_parquet(DATA_PATH)
    regime_df = pd.read_parquet(REGIME_DATA_PATH)

    master_df.reset_index(inplace=True)
    master_df.drop_duplicates(subset=['datetime', 'symbol'], keep='first', inplace=True)
    master_df.set_index('datetime', inplace=True)

    merged_df = pd.merge_asof(master_df.sort_index(), regime_df.sort_index(), left_index=True, right_index=True, direction='backward')
    
    start_ts = INDIA_TZ.localize(pd.to_datetime(START_DATE))
    end_ts = INDIA_TZ.localize(pd.to_datetime(END_DATE))
    merged_df = merged_df[(merged_df.index >= start_ts) & (merged_df.index <= end_ts)]
    
    unique_timestamps = sorted(merged_df.index.unique())
    if not unique_timestamps: print("No data in date range."); return

    cash, equity = INITIAL_CAPITAL, INITIAL_CAPITAL
    open_positions, closed_trades_log = [], []
    last_known_prices = {}
    equity_curve = [{'datetime': unique_timestamps[0], 'equity': INITIAL_CAPITAL}] # Initialize equity curve

    total_timestamps = len(unique_timestamps) # Get total for progress bar
    for i, ts in enumerate(unique_timestamps):
        current_data_slice = merged_df.loc[ts]
        if isinstance(current_data_slice, pd.Series):
             current_data_slice = current_data_slice.to_frame().T
        
        new_prices = current_data_slice.set_index('symbol').to_dict('index')
        last_known_prices.update(new_prices)
        
        if i == 0 or ts.date() != unique_timestamps[i-1].date():
            update_progress(i + 1, total_timestamps, ts, equity, open_positions)

        positions_to_close = []
        for trade in open_positions:
            candle = last_known_prices.get(trade['symbol'])
            if candle is None: continue

            exit_reason, exit_price = None, None
            
            if ts.time() == datetime.time(9, 15) and trade['entry_time'].date() < ts.date():
                if candle['open'] >= trade['sl']:
                    exit_reason, exit_price = 'GAP_SL_HIT', trade['sl']
                elif candle['open'] <= trade['tp']:
                    exit_reason, exit_price = 'GAP_TP_HIT', trade['tp']
            
            if not exit_reason:
                if candle['high'] >= trade['sl']:
                    exit_reason, exit_price = 'SL_HIT', trade['sl']
                elif candle['low'] <= trade['tp']:
                    exit_reason, exit_price = 'TP_HIT', trade['tp']
            
            if USE_BREAKEVEN_STOP and not trade['be_activated']:
                if candle['low'] <= trade['be_trigger_price']:
                    trade['sl'] = trade['be_target_price']
                    trade['be_activated'] = True
            
            current_profit_r = (trade['entry_price'] - candle['low']) / trade['initial_risk_per_share'] if trade['initial_risk_per_share'] > 0 else 0
            if USE_MULTI_STAGE_TS and current_profit_r >= AGGRESSIVE_TS_TRIGGER_R:
                trade['current_ts_multiplier'] = AGGRESSIVE_TS_MULTIPLIER

            if USE_ATR_TRAILING_STOP and pd.notna(candle.get('atr_14')):
                new_trailing_stop = candle['low'] + (candle['atr_14'] * trade['current_ts_multiplier'])
                trade['sl'] = min(trade['sl'], new_trailing_stop)

            if EXIT_ON_EOD and ts.strftime('%H:%M') == EOD_TIME and not exit_reason:
                if ALLOW_AFTERNOON_POSITIONAL and trade.get('is_afternoon_entry', False):
                    continue
                exit_reason, exit_price = 'EOD_EXIT', candle['close']

            if exit_reason:
                # BUG FIX v4: Implement correct cash flow for closing a short position
                cost_to_cover = (trade['quantity'] * exit_price) * (1 + TRANSACTION_COST_PCT/100)
                cash -= cost_to_cover
                
                # BUG FIX v4: PnL is the initial cash received minus the cash paid to cover
                pnl = trade['initial_proceeds'] - cost_to_cover
                
                trade.update({'exit_time': ts, 'exit_price': exit_price, 'exit_reason': exit_reason, 'pnl': pnl})
                closed_trades_log.append(trade)
                positions_to_close.append(trade)

        open_positions = [p for p in open_positions if p not in positions_to_close]

        today_regime = current_data_slice.iloc[0]
        regime_ok = (not USE_BREADTH_FILTER or today_regime.get('breadth_pct_below_sma', 0) > BREADTH_THRESHOLD_PCT)
        
        current_time_str = ts.strftime('%H:%M')
        if AVOID_OPEN_CLOSE_ENTRIES and (current_time_str == '09:15' or current_time_str == '15:15'):
            pass
        elif regime_ok and len(open_positions) < STRICT_MAX_OPEN_POSITIONS:
            potential_trades = current_data_slice[current_data_slice.get('is_entry_signal', False) == True]
            active_symbols = [p['symbol'] for p in open_positions]

            for _, signal in potential_trades.iterrows():
                if signal['symbol'] in active_symbols: continue

                entry_price_base = signal['alert_candle_low']
                entry_price = entry_price_base * (1 - SLIPPAGE_PCT/100)
                initial_sl = signal['pattern_high_for_sl']
                risk_per_share = initial_sl - entry_price
                if risk_per_share <= 0: continue

                risk_amount = min(equity * RISK_PER_TRADE_PCT, MAX_RISK_PER_TRADE_CAP)
                quantity = int(risk_amount / risk_per_share)
                
                if quantity > 0:
                    # BUG FIX v4: Implement correct cash flow for opening a short position
                    initial_proceeds = (quantity * entry_price) * (1 - TRANSACTION_COST_PCT/100)
                    cash += initial_proceeds
                    
                    is_afternoon_entry = ts.strftime('%H:%M') >= AFTERNOON_ENTRY_THRESHOLD

                    new_trade = {
                        'symbol': signal['symbol'], 
                        'direction': 'SHORT',
                        'entry_time': ts, 
                        'entry_price': entry_price, 
                        'sl': initial_sl, 
                        'tp': entry_price - (risk_per_share * RISK_REWARD_RATIO),
                        'quantity': quantity,
                        'initial_sl': initial_sl,
                        'initial_risk_per_share': risk_per_share,
                        'be_activated': False,
                        'be_trigger_price': entry_price - (risk_per_share * BREAKEVEN_TRIGGER_R),
                        'be_target_price': entry_price - (risk_per_share * BREAKEVEN_PROFIT_R),
                        'current_ts_multiplier': ATR_TS_MULTIPLIER,
                        'is_afternoon_entry': is_afternoon_entry,
                        'initial_proceeds': initial_proceeds # Store for accurate P&L calculation
                    }
                    open_positions.append(new_trade)
        
        market_value_of_shorts = 0
        for trade in open_positions:
            last_price = last_known_prices.get(trade['symbol'])
            if last_price:
                market_value_of_shorts += trade['quantity'] * last_price['close']
        equity = cash - market_value_of_shorts
        equity_curve.append({'datetime': ts, 'equity': equity})

    update_progress(total_timestamps, total_timestamps, unique_timestamps[-1], equity, open_positions)
    print("\n\n--- Backtest Finished ---")
    
    closed_df = pd.DataFrame(closed_trades_log)
    if not closed_df.empty: closed_df.to_csv(os.path.join(log_dir, 'trade_log.csv'), index=False)
    
    equity_df = pd.DataFrame(equity_curve).set_index('datetime')
    log_summary(log_dir, closed_df, equity_df)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    equity_df['equity'].plot(figsize=(12, 8), title='Portfolio Equity Curve')
    plt.ylabel('Equity')
    plt.xlabel('Date')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'equity_curve.png'))

    print(f"Results saved to: {log_dir}")

if __name__ == "__main__":
    main()

