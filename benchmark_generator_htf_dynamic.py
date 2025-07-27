# benchmark_generator_htf_dynamic.py
#
# Description:
# This is a new benchmark generator built to test a different HTF strategy vision.
# It identifies a weekly pullback and then monitors the daily chart for a
# "staging" signal (a green candle) before a final price breakout.
#
# MODIFICATION:
# 1. ENHANCEMENT: Upgraded the logging and final reporting to match the
#    comprehensive standard of the other benchmark scripts in the project.
#    This includes a detailed summary, a trades log, and an all_setups_log
#    with hypothetical performance simulation.

import pandas as pd
import os
import math
from datetime import datetime, timedelta
import time
import sys

# --- CONFIGURATION ---
config = {
    'initial_capital': 1000000,
    'risk_per_trade_percent': 2.0,
    'timeframe': 'weekly-immediate',
    'data_folder_base': 'data/processed',
    'log_folder': 'backtest_logs',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',
    
    # --- CORE STRATEGY PARAMETERS ---
    'ema_period': 30,
    'stop_loss_lookback': 5,
    'rs_period': 30,
    
    # --- EOD FILTERS ---
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200_INDEX',
    'regime_ma_period': 50,
    'volume_filter': True,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    'rs_filter': True,
    'use_partial_profit_leg': True,
}

def get_consecutive_red_candles_htf(df, current_loc):
    """Finds all consecutive red candles on the HTF chart before a certain point."""
    red_candles = []
    i = current_loc - 1
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def simulate_trade_outcome(symbol, entry_date, entry_price, stop_loss, daily_data, cfg):
    df = daily_data[symbol]
    target_price = entry_price + (entry_price - stop_loss)
    current_stop = stop_loss
    partial_exit_pnl, final_pnl, leg1_sold, exit_date = 0, 0, False, None
    trade_dates = df.loc[entry_date:].index[1:]
    for date in trade_dates:
        if date not in df.index: continue
        candle = df.loc[date]
        if cfg['use_partial_profit_leg'] and not leg1_sold and candle['high'] >= target_price:
            partial_exit_pnl = target_price - entry_price; leg1_sold = True; current_stop = entry_price
        if candle['low'] <= current_stop:
            final_pnl = current_stop - entry_price; exit_date = date; break
        if candle['close'] > entry_price:
            new_stop = max(current_stop, entry_price)
            if candle['close'] > candle['open']: new_stop = max(new_stop, candle['low'])
            current_stop = new_stop
    if exit_date is None: exit_date, final_pnl = df.index[-1], df.iloc[-1]['close'] - entry_price
    total_pnl = (partial_exit_pnl * 0.5) + (final_pnl * 0.5) if leg1_sold else final_pnl
    return exit_date, total_pnl

def run_backtest(cfg):
    start_time = time.time()
    base_timeframe = cfg['timeframe'].replace('-immediate', '')
    data_folder_htf = os.path.join(cfg['data_folder_base'], base_timeframe)
    data_folder_daily = os.path.join(cfg['data_folder_base'], 'daily')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(cfg['log_folder'], exist_ok=True)
    
    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return

    index_df_daily = None
    try:
        index_path = os.path.join(data_folder_daily, f"{cfg['regime_index_symbol']}_daily_with_indicators.csv")
        index_df_daily = pd.read_csv(index_path, index_col='datetime', parse_dates=True)
        index_df_daily.rename(columns=lambda x: x.lower(), inplace=True)
    except FileNotFoundError:
        cfg['market_regime_filter'], cfg['rs_filter'] = False, False

    stock_data_daily, stock_data_htf = {}, {}
    for symbol in symbols:
        daily_path = os.path.join(data_folder_daily, f"{symbol}_daily_with_indicators.csv")
        if os.path.exists(daily_path):
            df_d = pd.read_csv(daily_path, index_col='datetime', parse_dates=True)
            df_d.rename(columns=lambda x: x.lower(), inplace=True)
            df_d['green_candle'] = df_d['close'] > df_d['open']
            stock_data_daily[symbol] = df_d
        htf_path = os.path.join(data_folder_htf, f"{symbol}_{base_timeframe}_with_indicators.csv")
        if os.path.exists(htf_path):
            df_h = pd.read_csv(htf_path, index_col='datetime', parse_dates=True)
            df_h.rename(columns=lambda x: x.lower(), inplace=True)
            if not df_h.empty:
                df_h['red_candle'] = df_h['close'] < df_h['open']
                stock_data_htf[symbol] = df_h

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    all_setups_log = []
    monitoring_list = {}
    
    all_dates = index_df_daily.loc[cfg['start_date']:cfg['end_date']].index
    
    print("Starting DYNAMIC HTF Benchmark Generator...")
    for date in all_dates:
        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Monitoring: {len(monitoring_list)}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()
        
        exit_proceeds, to_remove, todays_exits = 0, [], []
        for pos_id, pos in list(portfolio['positions'].items()):
            symbol = pos['symbol']
            if symbol not in stock_data_daily or date not in stock_data_daily[symbol].index: continue
            daily_candle = stock_data_daily[symbol].loc[date]
            if cfg['use_partial_profit_leg'] and not pos.get('partial_exit', False) and daily_candle['high'] >= pos['target']:
                shares, price = pos['shares'] // 2, pos['target']
                exit_proceeds += shares * price
                todays_exits.append({'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date, 'pnl': (price - pos['entry_price']) * shares, 'exit_type': 'Partial Profit (1:1)', **pos})
                pos['shares'] -= shares; pos['partial_exit'] = True; pos['stop_loss'] = pos['entry_price']
            if pos['shares'] > 0 and daily_candle['low'] <= pos['stop_loss']:
                price, exit_type = pos['stop_loss'], 'Stop-Loss'
                exit_proceeds += pos['shares'] * price
                todays_exits.append({'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date, 'pnl': (price - pos['entry_price']) * pos['shares'], 'exit_type': exit_type, **pos})
                to_remove.append(pos_id); continue
            if pos['shares'] > 0 and daily_candle['close'] > pos['entry_price']:
                new_stop = max(pos['stop_loss'], pos['entry_price'])
                if daily_candle['close'] > daily_candle['open']: new_stop = max(new_stop, daily_candle['low'])
                pos['stop_loss'] = new_stop
        for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
        portfolio['cash'] += exit_proceeds
        portfolio['trades'].extend(todays_exits)
        
        if date.weekday() == 0:
            for symbol, df_h in stock_data_htf.items():
                try:
                    htf_loc = df_h.index.searchsorted(date, side='right') - 1
                    if htf_loc < 1: continue
                    red_candles = get_consecutive_red_candles_htf(df_h, htf_loc)
                    if red_candles:
                        trigger_price = max([c['high'] for c in red_candles])
                        monitoring_list[symbol] = {'trigger_price': trigger_price, 'staged': False, 'setup_date': red_candles[0].name}
                except (KeyError, IndexError): continue
        
        market_uptrend = True
        if cfg['market_regime_filter'] and date in index_df_daily.index:
            if index_df_daily.loc[date]['close'] < index_df_daily.loc[date][f"ema_{cfg['regime_ma_period']}"]:
                market_uptrend = False
        
        if market_uptrend:
            for symbol, details in list(monitoring_list.items()):
                if any(pos['symbol'] == symbol for pos in portfolio['positions'].values()):
                    del monitoring_list[symbol]; continue
                if symbol not in stock_data_daily or date not in stock_data_daily[symbol].index: continue
                
                df_d = stock_data_daily[symbol]
                today_candle = df_d.loc[date]
                trigger_price = details['trigger_price']
                
                if not details['staged'] and today_candle['green_candle'] and today_candle['high'] < trigger_price:
                    details['staged'] = True
                
                if details['staged'] and today_candle['high'] >= trigger_price:
                    log_entry = {'symbol': symbol, 'setup_date': details['setup_date'], 'trigger_date': date, 'trigger_price': trigger_price, 'status': 'IDENTIFIED'}
                    
                    rs_ok = not cfg['rs_filter'] or (today_candle[f"return_{cfg['rs_period']}"] > index_df_daily.loc[date, f"return_{cfg['rs_period']}"])
                    if not rs_ok: log_entry['status'] = 'FILTERED_RS'; all_setups_log.append(log_entry); del monitoring_list[symbol]; continue
                    
                    volume_ok = not cfg['volume_filter'] or (pd.notna(today_candle[f"volume_{cfg['volume_ma_period']}_sma"]) and today_candle['volume'] >= today_candle[f"volume_{cfg['volume_ma_period']}_sma"] * cfg['volume_multiplier'])
                    if not volume_ok: log_entry['status'] = 'FILTERED_VOLUME'; all_setups_log.append(log_entry); del monitoring_list[symbol]; continue
                    
                    entry_price = trigger_price
                    loc_d = df_d.index.get_loc(date)
                    stop_loss = df_d.iloc[max(0, loc_d - cfg['stop_loss_lookback']):loc_d]['low'].min()
                    if pd.isna(stop_loss): stop_loss = df_d.iloc[loc_d-1]['low']
                    
                    risk_per_share = entry_price - stop_loss
                    if risk_per_share > 0:
                        shares = math.floor((portfolio['equity'] * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)
                        if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                            log_entry['status'] = 'FILLED'; portfolio['cash'] -= shares * entry_price
                            portfolio['positions'][f"{symbol}_{date}"] = {'symbol': symbol, 'entry_date': date, 'entry_price': entry_price, 'stop_loss': stop_loss, 'shares': shares, 'target': entry_price + risk_per_share, 'partial_exit': False, 'initial_shares': shares}
                        elif shares > 0: log_entry['status'] = 'MISSED_CAPITAL'
                        all_setups_log.append(log_entry)
                    
                    del monitoring_list[symbol]

        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in stock_data_daily.get(pos['symbol'], pd.DataFrame()).index:
                eod_equity += pos['shares'] * stock_data_daily[pos['symbol']].loc[date]['close']
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    print("\n\n--- DYNAMIC HTF BENCHMARK COMPLETE ---")
    all_setups_df = pd.DataFrame(all_setups_log)
    for index, log_entry in all_setups_df[all_setups_df['status'] == 'MISSED_CAPITAL'].iterrows():
        if log_entry['trigger_date'] in stock_data_daily[log_entry['symbol']].index:
            loc_d = stock_data_daily[log_entry['symbol']].index.get_loc(log_entry['trigger_date'])
            stop_loss = stock_data_daily[log_entry['symbol']].iloc[max(0, loc_d - cfg['stop_loss_lookback']):loc_d]['low'].min()
            if pd.notna(stop_loss):
                exit_date, pnl = simulate_trade_outcome(log_entry['symbol'], log_entry['trigger_date'], log_entry['trigger_price'], stop_loss, stock_data_daily, cfg)
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
    summary_content = f"""BACKTEST SUMMARY REPORT (DYNAMIC HTF BENCHMARK)\n================================\n{params_str}\nREALISTIC PERFORMANCE (CAPITAL CONSTRAINED):\n--------------------------------------------\nFinal Equity: {final_equity:,.2f}\nNet P&L: {net_pnl:,.2f}\nCAGR: {cagr:.2f}%\nMax Drawdown: {max_drawdown:.1f}%\nTotal Trade Events (incl. partials): {total_trades}\nWin Rate (of events): {win_rate:.1f}%\nProfit Factor: {profit_factor:.2f}\n\nHYPOTHETICAL PERFORMANCE (UNCONSTRAINED):\n-----------------------------------------\nTotal Setups Found: {len(all_setups_df[all_setups_df['status'].isin(['FILLED', 'MISSED_CAPITAL'])])}\nStrategy Win Rate (per setup): {hypothetical_win_rate:.1f}%\nStrategy Profit Factor (per setup): {hypothetical_profit_factor:.2f}\n"""
    
    summary_filename = os.path.join(cfg['log_folder'], f"{timestamp}_summary_report_htf_dynamic.txt")
    trades_filename = os.path.join(cfg['log_folder'], f"{timestamp}_trades_detail_htf_dynamic.csv")
    all_setups_filename = os.path.join(cfg['log_folder'], f"{timestamp}_all_setups_log_htf_dynamic.csv")
    
    with open(summary_filename, 'w') as f: f.write(summary_content)
    if not trades_df.empty: trades_df.to_csv(trades_filename, index=False)
    if not all_setups_df.empty: all_setups_df.to_csv(all_setups_filename, index=False)
    print(summary_content)
    
if __name__ == "__main__":
    run_backtest(config)
