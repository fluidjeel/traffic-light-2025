# benchmark_generator_daily.py
#
# Description:
# This script creates the "Golden Benchmark" for the daily strategy.
#
# MODIFICATION (v2.0 - Logging Enhancement):
# 1. ADDED: A dedicated subdirectory ('benchmark_daily') is now created under
#    the main log folder for organized report storage.
# 2. UPDATED: The script configuration now includes a 'strategy_name'
#    to dynamically create the log folder.

import pandas as pd
import os
import math
from datetime import datetime
import time
import sys

# --- CONFIGURATION ---
config = {
    'initial_capital': 1000000,
    'risk_per_trade_percent': 4.0,
    'timeframe': 'daily', 
    'data_folder_base': 'data/processed',
    'log_folder': 'backtest_logs',
    'strategy_name': 'benchmark_daily', # For dedicated log folder
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',
    'ema_period': 30,
    'stop_loss_lookback': 5,
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200_INDEX',
    'regime_ma_period': 50,
    'volume_filter': True,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    'rs_filter': True,
    'rs_index_symbol': 'NIFTY200_INDEX',
    'rs_period': 30,
    'use_partial_profit_leg': True,
}

def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 2 
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def simulate_trade_outcome(symbol, entry_date, entry_price, stop_loss, daily_data):
    df = daily_data[symbol]
    target_price = entry_price + (entry_price - stop_loss)
    current_stop = stop_loss
    partial_exit_pnl, final_pnl, leg1_sold, exit_date = 0, 0, False, None
    trade_dates = df.loc[entry_date:].index[1:]
    for date in trade_dates:
        if date not in df.index: continue
        candle = df.loc[date]
        if not leg1_sold and candle['high'] >= target_price:
            partial_exit_pnl = target_price - entry_price; leg1_sold = True; current_stop = entry_price
        if candle['low'] <= current_stop:
            final_pnl = current_stop - entry_price; exit_date = date; break
        if candle['close'] > entry_price:
            current_stop = max(current_stop, entry_price)
            if candle['green_candle']: current_stop = max(current_stop, candle['low'])
    if exit_date is None: exit_date, final_pnl = df.index[-1], df.iloc[-1]['close'] - entry_price
    total_pnl = (partial_exit_pnl * 0.5) + (final_pnl * 0.5) if leg1_sold else final_pnl
    return exit_date, total_pnl

def run_backtest(cfg):
    start_time = time.time()
    data_folder = os.path.join(cfg['data_folder_base'], cfg['timeframe'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # --- ENHANCEMENT 1: Create dedicated log folder ---
    strategy_log_folder = os.path.join(cfg['log_folder'], cfg['strategy_name'])
    os.makedirs(strategy_log_folder, exist_ok=True)
    
    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return

    index_df_daily = None
    all_dates = []
    try:
        index_path = os.path.join(data_folder, f"{cfg['regime_index_symbol']}_daily_with_indicators.csv")
        index_df_daily = pd.read_csv(index_path, index_col='datetime', parse_dates=True)
        all_dates = index_df_daily.loc[cfg['start_date']:cfg['end_date']].index
    except FileNotFoundError:
        print(f"Error: Index file not found at {index_path}. Disabling filters.")
        cfg['market_regime_filter'], cfg['rs_filter'] = False, False

    stock_data = {}
    print(f"Loading and preprocessing data from '{data_folder}'...")
    for symbol in symbols + [cfg['regime_index_symbol']]:
        file_path = os.path.join(data_folder, f"{symbol}_daily_with_indicators.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
                df = df.loc[cfg['start_date']:cfg['end_date']]
                if not df.empty:
                    df['red_candle'] = df['close'] < df['open']
                    df['green_candle'] = df['close'] > df['open']
                    stock_data[symbol] = df
            except Exception as e: print(f"Error processing {symbol}: {str(e)}")
    print(f"Successfully processed data for {len(stock_data)} symbols.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    all_setups_log = []
    
    print("Starting Daily Benchmark Generator...")
    for date in all_dates:
        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()
        
        exit_proceeds, to_remove, todays_exits = 0, [], []
        for pos_id, pos in list(portfolio['positions'].items()):
            symbol = pos['symbol']
            if date not in stock_data[symbol].index: continue
            candle = stock_data[symbol].loc[date]
            if cfg['use_partial_profit_leg'] and not pos.get('partial_exit', False) and candle['high'] >= pos['target']:
                shares, price = pos['shares'] // 2, pos['target']
                exit_proceeds += shares * price
                todays_exits.append({'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date, 'pnl': (price - pos['entry_price']) * shares, 'exit_type': 'Partial Profit (1:1)', **pos})
                pos['shares'] -= shares; pos['partial_exit'] = True; pos['stop_loss'] = pos['entry_price']
            if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                price, exit_type = pos['stop_loss'], 'Stop-Loss'
                exit_proceeds += pos['shares'] * price
                todays_exits.append({'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date, 'pnl': (price - pos['entry_price']) * pos['shares'], 'exit_type': exit_type, **pos})
                to_remove.append(pos_id); continue
            if pos['shares'] > 0 and candle['close'] > pos['entry_price']:
                pos['stop_loss'] = max(pos['stop_loss'], pos['entry_price'])
                if candle['green_candle']: pos['stop_loss'] = max(pos['stop_loss'], candle['low'])
        for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
        portfolio['cash'] += exit_proceeds
        portfolio['trades'].extend(todays_exits)

        market_uptrend = True
        if cfg['market_regime_filter'] and date in stock_data[cfg['regime_index_symbol']].index:
            if stock_data[cfg['regime_index_symbol']].loc[date]['close'] < stock_data[cfg['regime_index_symbol']].loc[date][f"ema_{cfg['regime_ma_period']}"]:
                market_uptrend = False
        
        if market_uptrend:
            for symbol, df in stock_data.items():
                if any(pos['symbol'] == symbol for pos in portfolio['positions'].values()): continue
                if date not in df.index: continue
                try:
                    loc = df.index.get_loc(date)
                    if loc < 2: continue
                    setup_candle = df.iloc[loc-1]
                    setup_date = setup_candle.name
                    
                    if not setup_candle['green_candle']: continue
                        
                    if not (setup_candle['close'] > setup_candle[f"ema_{cfg['ema_period']}"]): continue
                    red_candles = get_consecutive_red_candles(df, loc)
                    if not red_candles: continue
                    trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])
                    trigger_candle = df.iloc[loc]
                    if not (trigger_candle['high'] >= trigger_price and trigger_candle['open'] < trigger_price): continue
                    
                    setup_id = f"{symbol}_{setup_date.strftime('%Y-%m-%d')}"
                    log_entry = {'setup_id': setup_id, 'symbol': symbol, 'setup_date': setup_date, 'trigger_date': date, 'trigger_price': trigger_price, 'status': 'IDENTIFIED'}
                    
                    volume_ok = not cfg['volume_filter'] or (trigger_candle['volume'] > trigger_candle[f"volume_{cfg['volume_ma_period']}_sma"] * cfg['volume_multiplier'])
                    if not volume_ok: log_entry['status'] = 'FILTERED_VOLUME'; all_setups_log.append(log_entry); continue
                    rs_ok = not cfg['rs_filter'] or (trigger_candle[f"return_{cfg['rs_period']}"] > stock_data[cfg['rs_index_symbol']].loc[date][f"return_{cfg['rs_period']}"])
                    if not rs_ok: log_entry['status'] = 'FILTERED_RS'; all_setups_log.append(log_entry); continue

                    entry_price = trigger_price
                    stop_loss = df.iloc[max(0, loc - cfg['stop_loss_lookback']):loc]['low'].min()
                    if pd.isna(stop_loss): stop_loss = df.iloc[loc - 1]['low']
                    risk_per_share = entry_price - stop_loss
                    if risk_per_share <= 0: continue
                    shares = math.floor((portfolio['equity'] * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)
                    if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                        log_entry['status'] = 'FILLED'; portfolio['cash'] -= shares * entry_price
                        portfolio['positions'][f"{symbol}_{date}"] = {'symbol': symbol, 'entry_date': date, 'entry_price': entry_price, 'stop_loss': stop_loss, 'shares': shares, 'target': entry_price + risk_per_share, 'initial_shares': shares, 'setup_id': setup_id}
                    elif shares > 0: log_entry['status'] = 'MISSED_CAPITAL'
                    all_setups_log.append(log_entry)
                except (KeyError, IndexError): continue
        
        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in stock_data.get(pos['symbol'], pd.DataFrame()).index:
                eod_equity += pos['shares'] * stock_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    print("\n\n--- BACKTEST COMPLETE ---")
    all_setups_df = pd.DataFrame(all_setups_log)
    for index, log_entry in all_setups_df[all_setups_df['status'] != 'FILLED'].iterrows():
        df = stock_data[log_entry['symbol']]
        loc = df.index.get_loc(log_entry['trigger_date'])
        stop_loss = df.iloc[max(0, loc - cfg['stop_loss_lookback']):loc]['low'].min()
        if pd.isna(stop_loss): stop_loss = df.iloc[loc - 1]['low']
        if pd.notna(stop_loss):
            exit_date, pnl = simulate_trade_outcome(log_entry['symbol'], log_entry['trigger_date'], log_entry['trigger_price'], stop_loss, stock_data)
            all_setups_df.loc[index, 'hypothetical_exit_date'] = exit_date
            all_setups_df.loc[index, 'hypothetical_pnl_per_share'] = pnl

    final_equity = portfolio['equity']
    net_pnl = final_equity - cfg['initial_capital']
    equity_df = pd.DataFrame(portfolio['daily_values']).set_index('date')
    if not equity_df.empty:
        years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = ((final_equity / cfg['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
        peak = equity_df['equity'].cummax(); drawdown = (equity_df['equity'] - peak) / peak; max_drawdown = abs(drawdown.min()) * 100
    else: cagr, max_drawdown = 0, 0
    trades_df = pd.DataFrame(portfolio['trades'])
    total_trades, win_rate, profit_factor, avg_win, avg_loss = 0, 0, 0, 0, 0
    if not trades_df.empty:
        winning_trades, losing_trades = trades_df[trades_df['pnl'] > 0], trades_df[trades_df['pnl'] <= 0]
        total_trades = len(trades_df); win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        gross_profit, gross_loss = winning_trades['pnl'].sum(), abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win, avg_loss = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0, abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
    hypothetical_win_rate, hypothetical_profit_factor = 0, 0
    if not all_setups_df.empty:
        filled_trades_for_hypo = trades_df[trades_df['exit_type'] != 'Partial Profit (1:1)'].copy()
        if 'initial_shares' in filled_trades_for_hypo.columns and not filled_trades_for_hypo.empty:
            filled_trades_for_hypo = filled_trades_for_hypo[filled_trades_for_hypo['initial_shares'] > 0]
            filled_trades_for_hypo['pnl_per_share'] = filled_trades_for_hypo['pnl'] / filled_trades_for_hypo['initial_shares']
        else: filled_trades_for_hypo['pnl_per_share'] = 0
        hypo_pnl_list = list(all_setups_df[all_setups_df['status'] == 'MISSED_CAPITAL'].get('hypothetical_pnl_per_share', []).dropna()) + list(filled_trades_for_hypo['pnl_per_share'].dropna())
        if hypo_pnl_list:
            winning_setups, losing_setups = [p for p in hypo_pnl_list if p > 0], [p for p in hypo_pnl_list if p <= 0]
            hypothetical_win_rate = (len(winning_setups) / len(hypo_pnl_list)) * 100 if hypo_pnl_list else 0
            gross_hypo_profit, gross_hypo_loss = sum(winning_setups), abs(sum(losing_setups))
            hypothetical_profit_factor = gross_hypo_profit / gross_hypo_loss if gross_hypo_loss > 0 else float('inf')

    params_str = "INPUT PARAMETERS:\n-----------------\n"
    for key, value in cfg.items(): params_str += f"{key.replace('_', ' ').title()}: {value}\n"
    summary_content = f"""BACKTEST SUMMARY REPORT (DAILY BENCHMARK)
================================
{params_str}
REALISTIC PERFORMANCE (CAPITAL CONSTRAINED):
--------------------------------------------
Final Equity: {final_equity:,.2f}
Net P&L: {net_pnl:,.2f}
CAGR: {cagr:.2f}%
Max Drawdown: {max_drawdown:.1f}%
Total Trade Events (incl. partials): {total_trades}
Win Rate (of events): {win_rate:.1f}%
Profit Factor: {profit_factor:.2f}

HYPOTHETICAL PERFORMANCE (UNCONSTRAINED):
-----------------------------------------
Total Setups Found: {len(all_setups_df[all_setups_df['status'].isin(['FILLED', 'MISSED_CAPITAL'])])}
Strategy Win Rate (per setup): {hypothetical_win_rate:.1f}%
Strategy Profit Factor (per setup): {hypothetical_profit_factor:.2f}
"""
    
    # --- UPDATED: Use the new dedicated log folder ---
    summary_filename = os.path.join(strategy_log_folder, f"{timestamp}_summary_report.txt")
    trades_filename = os.path.join(strategy_log_folder, f"{timestamp}_trades_detail.csv")
    all_setups_filename = os.path.join(strategy_log_folder, f"{timestamp}_all_setups_log.csv")
    
    with open(summary_filename, 'w') as f: f.write(summary_content)
    if not trades_df.empty: trades_df.to_csv(trades_filename, index=False)
    if not all_setups_df.empty: all_setups_df.to_csv(all_setups_filename, index=False)
    print(summary_content)
    print(f"\nReports saved to '{strategy_log_folder}'")
    
if __name__ == "__main__":
    run_backtest(config)
