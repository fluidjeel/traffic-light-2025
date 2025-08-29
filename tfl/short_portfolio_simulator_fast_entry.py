# short_portfolio_simulator_fast_entry.py
#
# Description:
# A professional-grade, portfolio-level backtesting simulator designed for maximum realism.
# This version is specifically built for the "Fast Entry" model, where trades are
# executed on the same candle the entry signal is generated.
#
# FIX:
# - Corrected the EXIT_ON_EOD logic to ensure it functions correctly for intraday mode.
# - Re-introduced the MAX_RISK_PER_TRADE_CAP to prevent unrealistic compounding.

import os
import pandas as pd
import numpy as np
import datetime
from pytz import timezone
import sys

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# --- Backtest Period ---
START_DATE = '2018-01-01'
END_DATE = '2025-08-22'

# --- Portfolio & Risk Management ---
STRATEGY_NAME = "TrafficLight-Manny-SHORTS_PORTFOLIO_FAST_ENTRY"
INITIAL_CAPITAL = 1000000.00
RISK_PER_TRADE_PCT = 0.01
STRICT_MAX_OPEN_POSITIONS = 15
SLIPPAGE_PCT = 0.05
TRANSACTION_COST_PCT = 0.03 # Models brokerage, STT, fees, etc.
# Re-introduced to prevent runaway compounding
MAX_RISK_PER_TRADE_CAP = INITIAL_CAPITAL * 0.2 

# --- SIMULATOR MODE ---
EXIT_ON_EOD = False # Set to False to enable positional (overnight holding) mode

# --- MARKET REGIME FILTERS ---
USE_BREADTH_FILTER = True
BREADTH_THRESHOLD_PCT = 60.0 # Only trade if > 60% of stocks are below their 50-day SMA

USE_VOLATILITY_FILTER = True
VIX_THRESHOLD = 17.0 # Only trade if India VIX is > 17

TREND_FILTER_SMA_PERIOD = 100 # Options: 20, 30, 50, 100, 200, or 0 to disable

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
# --- SIMULATOR ENGINE ---
# ==============================================================================

def run_portfolio_simulation():
    """Main function to orchestrate the portfolio-level backtest."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(LOGS_BASE_DIR, STRATEGY_NAME, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_configs(log_dir)

    print("Loading FAST ENTRY master data file...")
    df = pd.read_parquet(DATA_PATH)

    print("Loading and merging market regime data...")
    regime_df = pd.read_parquet(REGIME_DATA_PATH)
    regime_df.dropna(inplace=True)

    df.set_index('datetime', inplace=True)
    df.index = pd.to_datetime(df.index).tz_convert(INDIA_TZ) if df.index.tz is not None else pd.to_datetime(df.index).tz_localize(INDIA_TZ)
    regime_df.index = pd.to_datetime(regime_df.index).tz_convert(INDIA_TZ) if regime_df.index.tz is not None else pd.to_datetime(regime_df.index).tz_localize(INDIA_TZ)
    
    df = pd.merge_asof(df.sort_index(), regime_df.sort_index(), left_index=True, right_index=True, direction='backward')
    
    start_dt = INDIA_TZ.localize(datetime.datetime.strptime(START_DATE, '%Y-%m-%d'))
    end_dt = INDIA_TZ.localize(datetime.datetime.strptime(END_DATE, '%Y-%m-%d'))
    df = df.loc[start_dt:end_dt]
    
    unique_timestamps = df.index.unique()
    print(f"Data loaded. Simulating from {unique_timestamps[0]} to {unique_timestamps[-1]}")

    portfolio = {
        'cash': INITIAL_CAPITAL, 'equity': INITIAL_CAPITAL, 'open_positions': {},
        'equity_curve': pd.Series([INITIAL_CAPITAL], index=[unique_timestamps[0]])
    }
    completed_trades, rejected_trades = [], []
    max_open_positions = 0

    print("Starting event-driven simulation...")
    for i, ts in enumerate(unique_timestamps):
        current_market_data = df.loc[ts]
        if isinstance(current_market_data, pd.Series):
            current_market_data = current_market_data.to_frame().T

        update_open_positions(portfolio, current_market_data, ts, completed_trades)
        portfolio['equity'] = calculate_portfolio_value(portfolio, current_market_data)
        scan_and_execute_new_trades(portfolio, current_market_data, ts, rejected_trades)

        max_open_positions = max(max_open_positions, len(portfolio['open_positions']))
        if ts.time() == datetime.time(15, 15):
            portfolio['equity_curve'][ts] = portfolio['equity']
        
        if i % 25 == 0 or i == len(unique_timestamps) - 1:
            progress_pct = (i + 1) / len(unique_timestamps)
            equity_str = f"₹{portfolio['equity']:,.2f}"
            current_pos_str = len(portfolio['open_positions'])
            output_str = (f"Date: {ts.strftime('%Y-%m-%d')} | Equity: {equity_str:<18} | "
                          f"Current Open Positions: {current_pos_str:<3} | Progress: [{'#' * int(progress_pct * 20):<20}] {progress_pct:.1%}")
            sys.stdout.write('\r' + output_str)
            sys.stdout.flush()

    print("\n\nSimulation complete. Generating analysis and logs...")
    trades_df = pd.DataFrame(completed_trades)
    rejected_df = pd.DataFrame(rejected_trades)
    if not trades_df.empty: trades_df.to_csv(os.path.join(log_dir, 'trade_log.csv'), index=False)
    if not rejected_df.empty: rejected_df.to_csv(os.path.join(log_dir, 'rejected_trades.csv'), index=False)
    generate_summary(log_dir, trades_df, portfolio['equity_curve'], max_open_positions)
    print(f"All logs saved to: {log_dir}")


def scan_and_execute_new_trades(portfolio, market_data, ts, rejected_trades):
    potential_trades = market_data[market_data['is_entry_signal']].copy()
    if potential_trades.empty: return

    potential_trades.sort_values(by='daily_rsi', ascending=True, inplace=True)
    slots_available = STRICT_MAX_OPEN_POSITIONS - len(portfolio['open_positions'])
    
    for _, signal in potential_trades.iterrows():
        symbol = signal['symbol']
        if symbol in portfolio['open_positions']:
            continue

        if USE_BREADTH_FILTER and signal.get('breadth_pct_below_sma', 100) < BREADTH_THRESHOLD_PCT:
            rejected_trades.append({'timestamp': ts, 'symbol': symbol, 'reason': 'Rejected by Breadth Filter'})
            continue
        if USE_VOLATILITY_FILTER and signal.get('india_vix_close', 0) < VIX_THRESHOLD:
            rejected_trades.append({'timestamp': ts, 'symbol': symbol, 'reason': 'Rejected by Volatility Filter'})
            continue
        
        if TREND_FILTER_SMA_PERIOD > 0:
            trend_col = f'is_nifty_below_sma_{TREND_FILTER_SMA_PERIOD}'
            if not signal.get(trend_col, False):
                rejected_trades.append({'timestamp': ts, 'symbol': symbol, 'reason': f'Rejected by NIFTY Trend Filter (SMA {TREND_FILTER_SMA_PERIOD})'})
                continue

        if slots_available > 0:
            entry_price = signal['alert_candle_low'] * (1 - SLIPPAGE_PCT / 100)
            sl_price = signal['pattern_high_for_sl']
            initial_risk_per_share = sl_price - entry_price

            if initial_risk_per_share <= 0: continue

            # --- Apply Risk Cap ---
            risk_amount_dynamic = portfolio['equity'] * RISK_PER_TRADE_PCT
            risk_amount = min(risk_amount_dynamic, MAX_RISK_PER_TRADE_CAP) # Cap the risk
            
            quantity = int(risk_amount / initial_risk_per_share)

            if quantity == 0: continue

            tp_price = entry_price - (initial_risk_per_share * RISK_REWARD_RATIO)
            
            portfolio['open_positions'][symbol] = {
                'symbol': symbol, 'direction': 'SHORT', 'entry_time': ts,
                'entry_price': entry_price, 'sl': sl_price, 'tp': tp_price,
                'quantity': quantity, 
                'initial_sl': sl_price, # Enhanced Logging
                'initial_risk_per_share': initial_risk_per_share,
                'be_activated': False,
                'be_trigger_price': entry_price - (initial_risk_per_share * BREAKEVEN_TRIGGER_R),
                'be_target_price': entry_price - (initial_risk_per_share * BREAKEVEN_PROFIT_R),
                'current_ts_multiplier': ATR_TS_MULTIPLIER
            }
            slots_available -= 1
        else:
            rejected_trades.append({'timestamp': ts, 'symbol': symbol, 'reason': 'Strict position limit reached'})


def update_open_positions(portfolio, market_data, ts, completed_trades):
    positions_to_close = []
    for symbol, trade in portfolio['open_positions'].items():
        if symbol not in market_data['symbol'].values: continue
        data = market_data[market_data['symbol'] == symbol].iloc[0]
        exit_reason, exit_price = None, None
        
        is_opening_candle = ts.time() == datetime.time(9, 15)
        if is_opening_candle and not EXIT_ON_EOD:
            gapped_past_sl = data['open'] >= trade['sl']
            gapped_past_tp = data['open'] <= trade['tp']
            
            if gapped_past_sl and gapped_past_tp:
                exit_reason, exit_price = 'GAP_SL_HIT', trade['sl']
            elif gapped_past_sl:
                exit_reason, exit_price = 'GAP_SL_HIT', data['open']
            elif gapped_past_tp:
                exit_reason, exit_price = 'GAP_TP_HIT', data['open']

        if not exit_reason:
            if data['high'] >= trade['sl']:
                exit_reason, exit_price = 'SL_HIT', trade['sl']
            elif data['low'] <= trade['tp']:
                exit_reason, exit_price = 'TP_HIT', trade['tp']
            # FIX: EOD exit check is now the last check
            elif EXIT_ON_EOD and ts.strftime('%H:%M') == EOD_TIME:
                exit_reason, exit_price = 'EOD_EXIT', data['close']
        
        if exit_price:
            exit_price *= (1 + SLIPPAGE_PCT / 100)
        
        if exit_reason:
            gross_pnl = (trade['entry_price'] - exit_price) * trade['quantity']
            turnover = (trade['entry_price'] + exit_price) * trade['quantity']
            costs = turnover * (TRANSACTION_COST_PCT / 100)
            net_pnl = gross_pnl - costs
            portfolio['cash'] += net_pnl
            trade.update({
                'exit_time': ts, 
                'exit_price': exit_price, 
                'exit_reason': exit_reason, 
                'pnl': net_pnl,
                'final_sl': trade['sl'] # Enhanced Logging
            })
            completed_trades.append(trade)
            positions_to_close.append(symbol)
        else:
            update_trailing_stops(trade, data)
            
    for symbol in positions_to_close:
        del portfolio['open_positions'][symbol]

def update_trailing_stops(trade, data):
    if USE_BREAKEVEN_STOP and not trade['be_activated']:
        if data['low'] <= trade['be_trigger_price']:
            trade['sl'] = trade['be_target_price']
            trade['be_activated'] = True
    if USE_MULTI_STAGE_TS and trade['initial_risk_per_share'] > 0:
        current_profit_r = (trade['entry_price'] - data['low']) / trade['initial_risk_per_share']
        if current_profit_r >= AGGRESSIVE_TS_TRIGGER_R:
            trade['current_ts_multiplier'] = AGGRESSIVE_TS_MULTIPLIER
    if USE_ATR_TRAILING_STOP and pd.notna(data['atr_14']):
        new_trailing_stop = data['low'] + (data['atr_14'] * trade['current_ts_multiplier'])
        trade['sl'] = min(trade['sl'], new_trailing_stop)

def calculate_portfolio_value(portfolio, market_data):
    unrealized_pnl = 0
    for symbol, trade in portfolio['open_positions'].items():
        if symbol in market_data['symbol'].values:
            current_price = market_data[market_data['symbol'] == symbol].iloc[0]['close']
            unrealized_pnl += (trade['entry_price'] - current_price) * trade['quantity']
    return portfolio['cash'] + unrealized_pnl

def generate_summary(log_dir, trades_df, equity_curve, max_open_positions):
    summary = ""
    if trades_df.empty:
        summary += "No trades were executed.\n"
    else:
        total_trades = len(trades_df)
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        net_pnl = trades_df['pnl'].sum()
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else float('inf')
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25 if len(equity_curve) > 1 else 0
        cagr = ((equity_curve.iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
        summary += f"""
        --- Backtest Summary: {STRATEGY_NAME} ---
        Period: {START_DATE} to {END_DATE}
        --- Performance Metrics ---
        Net PnL:                {net_pnl:,.2f}
        CAGR:                   {cagr:.2f}%
        Max Drawdown:           {max_drawdown:.2%}
        Profit Factor:          {profit_factor:.2f}
        Total Trades:           {total_trades}
        Win Rate:               {win_rate:.2f}%
        Max Concurrent Pos:     {max_open_positions}
        """
    summary += f"""
    --- Configuration ---
    Initial Capital:        {INITIAL_CAPITAL:,.2f}
    Risk Per Trade:         {RISK_PER_TRADE_PCT:.2%}
    Strict Max Positions:   {STRICT_MAX_OPEN_POSITIONS}
    Slippage:               {SLIPPAGE_PCT/100:.3%}
    Transaction Costs:      {TRANSACTION_COST_PCT/100:.3%}
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

if __name__ == "__main__":
    run_portfolio_simulation()
