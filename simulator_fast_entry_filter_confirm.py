# simulator_fast_entry_filter_confirm.py
#
# Description:
# Implements Strategy 1: Enters trades on intraday price and filter confirmation.
# At the penultimate 15-min candle of the day, it reconciles any trade opened
# today. If the trade is not on track to meet the benchmark's EOD criteria,
# it is managed proactively:
# - If profitable, SL is moved to breakeven to lock in gains.
# - If losing, the position is closed immediately to cut losses.
#
# This version incorporates all enhancements and bug fixes from Strategy 2.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time, timedelta
import time
import sys
import numpy as np

# --- CONFIGURATION ---
config = {
    'initial_capital': 1000000,
    'risk_per_trade_percent': 2.0,
    'timeframe': 'daily', 
    'data_folder_base': 'data/processed',
    'intraday_data_folder': 'historical_data_15min',
    'log_folder': 'backtest_logs',
    'simulator_log_folder': 'fast_entry_filter_confirm_logs',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',
    
    # --- CORE STRATEGY PARAMETERS ---
    'ema_period': 30,
    'stop_loss_lookback': 5,
    'rs_period': 30,
    
    # --- SIMULATOR-ONLY RULES ---
    'cancel_on_gap_up': True,
    'intraday_market_strength_filter': True, # ENABLED for entry
    'volume_velocity_filter': True,          # ENABLED for entry
    'volume_velocity_threshold_pct': 100.0,
    'use_partial_profit_leg': True,
    'use_aggressive_breakeven': True,
    'breakeven_buffer_points': 0.05,
    
    # --- EOD FILTERS (For watchlist generation) ---
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200_INDEX',
    'regime_ma_period': 50,
    'volume_filter': True,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    'rs_filter': True,

    # --- RECONCILIATION PARAMETERS ---
    'rogue_breakeven_buffer_percent': 0.1,
}


def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 1 
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def simulate_trade_outcome(symbol, entry_date, entry_price, stop_loss, daily_data, cfg):
    """
    Simulates the outcome of a hypothetical trade for logging purposes.
    """
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
            current_stop = max(current_stop, entry_price)
            if candle['green_candle']: current_stop = max(current_stop, candle['low'])
    if exit_date is None: exit_date, final_pnl = df.index[-1], df.iloc[-1]['close'] - entry_price
    total_pnl = (partial_exit_pnl * 0.5) + (final_pnl * 0.5) if leg1_sold else final_pnl
    return exit_date, total_pnl


def run_backtest(cfg):
    start_time = time.time()
    daily_folder = os.path.join(cfg['data_folder_base'], 'daily')
    intraday_folder = cfg['intraday_data_folder']
    
    # --- Use fixed log folder ---
    log_run_folder = os.path.join(cfg['log_folder'], cfg['simulator_log_folder'])
    os.makedirs(log_run_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return

    # --- ROBUST DATA LOADING LOOP ---
    print("Loading all Daily and 15-min data into memory...")
    daily_data, intraday_data = {}, {}
    index_df_daily = None
    symbols_to_load = symbols + [cfg['regime_index_symbol']]

    for symbol in symbols_to_load:
        # --- Load Daily Data ---
        try:
            daily_file = os.path.join(daily_folder, f"{symbol}_daily_with_indicators.csv")
            if os.path.exists(daily_file):
                df_d = pd.read_csv(daily_file, index_col='datetime', parse_dates=True)
                if not df_d.empty and all(col in df_d.columns for col in ['open', 'close']):
                    df_d.rename(columns=lambda x: x.lower(), inplace=True)
                    df_d['red_candle'] = df_d['close'] < df_d['open']
                    df_d['green_candle'] = df_d['close'] > df_d['open']
                    daily_data[symbol] = df_d
                    if cfg['regime_index_symbol'] in symbol:
                        index_df_daily = df_d
                else:
                    print(f"Warning: Daily data for {symbol} is empty or malformed. Skipping.")
        except Exception as e:
            print(f"Warning: Failed to load or process daily data for {symbol}. Error: {e}")

        # --- Load Intraday Data ---
        try:
            intraday_filename = f"{symbol}_15min.csv"
            if cfg['regime_index_symbol'] in symbol:
                intraday_filename = "NIFTY_200_15min.csv"
            intraday_file_path = os.path.join(intraday_folder, intraday_filename)
            
            if os.path.exists(intraday_file_path):
                df_i = pd.read_csv(intraday_file_path, index_col='datetime', parse_dates=True)
                if not df_i.empty:
                    intraday_data[symbol] = df_i
        except Exception as e:
            print(f"Warning: Failed to load or process intraday data for {symbol}. Error: {e}")
    
    print("Data loading complete.")
    
    if index_df_daily is None:
        print("CRITICAL ERROR: Index data could not be loaded. Cannot proceed.")
        return

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    all_setups_log = []
    watchlist = {}
    
    master_dates = index_df_daily.loc[cfg['start_date']:cfg['end_date']].index
    
    print("Starting Strategy 1: Fast Entry with Filter Confirmation...")
    for date in master_dates:
        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])} | Watchlist: {len(watchlist.get(date, {}))}"
        sys.stdout.write(f"\r{progress_str.ljust(120)}"); sys.stdout.flush()

        equity_at_sod = portfolio['equity']
        todays_watchlist = watchlist.get(date, {})
        
        # --- Intraday Loop ---
        try:
            today_intraday_candles = intraday_data.get(cfg['regime_index_symbol'], pd.DataFrame()).loc[date.date().strftime('%Y-%m-%d')]
        except KeyError: 
            today_intraday_candles = pd.DataFrame()

        if not today_intraday_candles.empty:
            intraday_cumulative_volumes = {}
            for s in todays_watchlist:
                if s in intraday_data:
                    try:
                        intraday_cumulative_volumes[s] = intraday_data[s].loc[date.date().strftime('%Y-%m-%d')]['volume'].cumsum()
                    except KeyError: continue

            penultimate_candle_time = None
            if len(today_intraday_candles.index) > 1:
                penultimate_candle_time = today_intraday_candles.index[-2]

            for candle_time in today_intraday_candles.index:
                # --- Standard Intraday Exit Logic ---
                exit_proceeds, to_remove = 0, []
                for pos_id, pos in list(portfolio['positions'].items()):
                    if pos['symbol'] not in intraday_data or candle_time not in intraday_data[pos['symbol']].index: continue
                    candle = intraday_data[pos['symbol']].loc[candle_time]
                    if cfg['use_partial_profit_leg'] and not pos['partial_exit'] and candle['high'] >= pos['target']:
                        shares, price = pos['shares'] // 2, pos['target']
                        exit_proceeds += shares * price
                        portfolio['trades'].append({'symbol': pos['symbol'], 'entry_date': pos['entry_date'], 'exit_date': candle_time, 'exit_price': price, 'pnl': (price - pos['entry_price']) * shares, 'exit_type': 'Partial Profit (1:1)', **pos})
                        pos['shares'] -= shares; pos['partial_exit'] = True; pos['stop_loss'] = pos['entry_price']
                    if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                        price, exit_type = pos['stop_loss'], 'Stop-Loss'
                        exit_proceeds += pos['shares'] * price
                        portfolio['trades'].append({'symbol': pos['symbol'], 'entry_date': pos['entry_date'], 'exit_date': candle_time, 'exit_price': price, 'pnl': (price - pos['entry_price']) * pos['shares'], 'exit_type': exit_type, **pos})
                        to_remove.append(pos_id)
                for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
                portfolio['cash'] += exit_proceeds

                # --- Intraday Entry Logic with Filters ---
                for symbol, details in list(todays_watchlist.items()):
                    if symbol not in intraday_data or candle_time not in intraday_data[symbol].index or any(p['symbol'] == symbol for p in portfolio['positions'].values()): continue
                    candle = intraday_data[symbol].loc[candle_time]
                    
                    if candle['close'] >= details['trigger_price']:
                        filters_passed = True
                        if cfg['volume_velocity_filter']:
                            cumulative_volume = intraday_cumulative_volumes.get(symbol, pd.Series()).get(candle_time, 0)
                            prev_day_loc = daily_data[symbol].index.get_loc(date) - 1
                            avg_daily_volume = daily_data[symbol].iloc[prev_day_loc][f"volume_{cfg['volume_ma_period']}_sma"]
                            if cumulative_volume < (avg_daily_volume * (cfg['volume_velocity_threshold_pct'] / 100)): filters_passed = False
                        
                        if filters_passed and cfg['intraday_market_strength_filter']:
                            nifty_day_candles = intraday_data[cfg['regime_index_symbol']].loc[date.date().strftime('%Y-%m-%d')]
                            if nifty_day_candles.loc[candle_time]['close'] < nifty_day_candles.iloc[0]['open']: filters_passed = False
                        
                        if filters_passed:
                            entry_price = candle['close']
                            daily_df, daily_loc = daily_data[symbol], daily_data[symbol].index.get_loc(date)
                            stop_loss = daily_df.iloc[max(0, daily_loc - cfg['stop_loss_lookback']):loc]['low'].min()
                            if pd.isna(stop_loss): continue
                            risk_per_share = entry_price - stop_loss
                            if risk_per_share <= 0: continue
                            shares = math.floor((equity_at_sod * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)
                            for log in all_setups_log:
                                if log['setup_id'] == details['setup_id']:
                                    if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                                        log['status'] = 'FILLED'; portfolio['cash'] -= shares * entry_price
                                        portfolio['positions'][f"{symbol}_{candle_time}"] = {
                                            'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price, 
                                            'stop_loss': stop_loss, 'shares': shares, 'target': entry_price + risk_per_share, 
                                            'partial_exit': False, 'initial_shares': shares, 'setup_id': details['setup_id'],
                                            'recon_managed': False 
                                        }
                                    elif shares > 0: log['status'] = 'MISSED_CAPITAL'
                            del todays_watchlist[symbol]

                # --- Reconciliation Logic at Penultimate Candle ---
                if penultimate_candle_time and candle_time == penultimate_candle_time:
                    exit_proceeds_recon, to_remove_recon = 0, []
                    for pos_id, pos in list(portfolio['positions'].items()):
                        if pos['entry_date'].date() != date.date(): continue
                        symbol = pos['symbol']
                        
                        is_benchmark_quality = True
                        try:
                            current_day_volume = intraday_cumulative_volumes.get(symbol, pd.Series()).get(candle_time, 0)
                            prev_day_loc = daily_data[symbol].index.get_loc(date) - 1
                            avg_daily_volume = daily_data[symbol].iloc[prev_day_loc][f"volume_{cfg['volume_ma_period']}_sma"]
                            if current_day_volume < (avg_daily_volume * cfg['volume_multiplier']):
                                is_benchmark_quality = False
                            
                            temp_stock_close = daily_data[symbol]['close'].loc[:date - timedelta(days=1)].append(pd.Series([intraday_data[symbol].loc[candle_time]['close']], index=[date]))
                            temp_index_close = index_df_daily['close'].loc[:date - timedelta(days=1)].append(pd.Series([intraday_data[cfg['regime_index_symbol']].loc[candle_time]['close']], index=[date]))
                            stock_rs = (temp_stock_close.pct_change(periods=cfg['rs_period']).iloc[-1]) * 100
                            index_rs = (temp_index_close.pct_change(periods=cfg['rs_period']).iloc[-1]) * 100
                            if stock_rs < index_rs:
                                is_benchmark_quality = False
                        except Exception:
                            is_benchmark_quality = False

                        if not is_benchmark_quality:
                            current_price = intraday_data[symbol].loc[candle_time]['close']
                            if current_price > pos['entry_price']:
                                pos['stop_loss'] = pos['entry_price'] * (1 + cfg['rogue_breakeven_buffer_percent'] / 100)
                                pos['recon_managed'] = True
                            else:
                                exit_price = current_price
                                exit_proceeds_recon += pos['shares'] * exit_price
                                portfolio['trades'].append({'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': candle_time, 'exit_price': exit_price, 'pnl': (exit_price - pos['entry_price']) * pos['shares'], 'exit_type': 'Recon-Close', **pos})
                                to_remove_recon.append(pos_id)
                    
                    for pos_id in to_remove_recon: portfolio['positions'].pop(pos_id, None)
                    portfolio['cash'] += exit_proceeds_recon

        # --- EOD Watchlist Generation (With Filters) ---
        market_uptrend = True
        if cfg['market_regime_filter'] and date in index_df_daily.index:
            if index_df_daily.loc[date]['close'] < index_df_daily.loc[date][f"ema_{cfg['regime_ma_period']}"]:
                market_uptrend = False
        if market_uptrend:
            for symbol in symbols:
                if symbol not in daily_data or date not in daily_data[symbol].index: continue
                df = daily_data[symbol]
                try:
                    loc = df.index.get_loc(date)
                    if loc < 2: continue
                    setup_candle = df.iloc[loc]
                    if not setup_candle['green_candle']: continue
                    if not (setup_candle['close'] > setup_candle[f"ema_{cfg['ema_period']}"]): continue
                    red_candles = get_consecutive_red_candles(df, loc)
                    if not red_candles: continue
                    
                    rs_ok = not cfg['rs_filter'] or (date in index_df_daily.index and pd.notna(df.loc[date, f"return_{cfg['rs_period']}"]) and pd.notna(index_df_daily.loc[date, f"return_{cfg['rs_period']}"]) and df.loc[date, f"return_{cfg['rs_period']}"] > index_df_daily.loc[date, f"return_{cfg['rs_period']}"])
                    if not rs_ok: continue
                    
                    volume_ok = not cfg['volume_filter'] or (pd.notna(setup_candle[f"volume_{cfg['volume_ma_period']}_sma"]) and setup_candle['volume'] >= (setup_candle[f"volume_{cfg['volume_ma_period']}_sma"] * cfg['volume_multiplier']))
                    if not volume_ok: continue

                    trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])
                    setup_id = f"{symbol}_{date.strftime('%Y-%m-%d')}"
                    all_setups_log.append({'setup_id': setup_id, 'symbol': symbol, 'setup_date': date, 'trigger_price': trigger_price, 'status': 'IDENTIFIED'})
                    next_day = date + timedelta(days=1)
                    if next_day in master_dates:
                        if next_day not in watchlist: watchlist[next_day] = {}
                        watchlist[next_day][symbol] = {'trigger_price': trigger_price, 'setup_id': setup_id}
                except (KeyError, IndexError): continue

        # --- EOD Position and Equity Management ---
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                daily_candle = daily_data[pos['symbol']].loc[date]
                new_stop = pos['stop_loss']
                if daily_candle['close'] > pos['entry_price']:
                    new_stop = max(new_stop, pos['entry_price'])
                    if cfg['use_aggressive_breakeven'] and not pos['partial_exit']:
                        new_stop = max(new_stop, pos['entry_price'] + cfg['breakeven_buffer_points'])
                    if daily_candle['green_candle']:
                        new_stop = max(new_stop, daily_candle['low'])
                pos['stop_loss'] = new_stop
        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                eod_equity += pos['shares'] * daily_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    # --- ENHANCED FINAL REPORTING ---
    print("\n\n--- BACKTEST COMPLETE ---")
    all_setups_df = pd.DataFrame(all_setups_log)
    trades_df = pd.DataFrame(portfolio['trades'])

    # --- Calculate Hypothetical Performance ---
    for index, log_entry in all_setups_df.iterrows():
        if log_entry['status'] in ['FILLED']: continue
        
        symbol = log_entry['symbol']
        entry_date = pd.to_datetime(log_entry['setup_date']) + timedelta(days=1)
        if entry_date in daily_data.get(symbol, pd.DataFrame()).index:
            try:
                loc = daily_data[symbol].index.get_loc(entry_date)
                stop_loss = daily_data[symbol].iloc[max(0, loc - cfg['stop_loss_lookback']):loc]['low'].min()
                if pd.notna(stop_loss):
                    exit_date, pnl = simulate_trade_outcome(symbol, entry_date, log_entry['trigger_price'], stop_loss, daily_data, cfg)
                    all_setups_df.loc[index, 'hypothetical_exit_date'] = exit_date
                    all_setups_df.loc[index, 'hypothetical_pnl_per_share'] = pnl
            except KeyError:
                continue

    final_equity = portfolio['equity']
    net_pnl = final_equity - cfg['initial_capital']
    equity_df = pd.DataFrame(portfolio['daily_values']).set_index('date')
    if not equity_df.empty:
        years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = ((final_equity / cfg['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100
    else:
        cagr, max_drawdown = 0, 0
    
    total_trades, win_rate, profit_factor, avg_win, avg_loss = 0, 0, 0, 0, 0
    if not trades_df.empty:
        core_trades_df = trades_df[~trades_df['exit_type'].str.startswith('Recon')]
        winning_trades = core_trades_df[core_trades_df['pnl'] > 0]
        losing_trades = core_trades_df[core_trades_df['pnl'] <= 0]
        total_trades = len(core_trades_df)
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0

    hypothetical_win_rate, hypothetical_profit_factor = 0, 0
    if not all_setups_df.empty:
        filled_trades_for_hypo = trades_df[trades_df['exit_type'] != 'Partial Profit (1:1)'].copy()
        if 'initial_shares' in filled_trades_for_hypo.columns and not filled_trades_for_hypo.empty:
             filled_trades_for_hypo = filled_trades_for_hypo[filled_trades_for_hypo['initial_shares'] > 0]
             filled_trades_for_hypo['pnl_per_share'] = filled_trades_for_hypo['pnl'] / filled_trades_for_hypo['initial_shares']
        else:
             filled_trades_for_hypo['pnl_per_share'] = 0
        
        hypo_pnl_list = list(all_setups_df[all_setups_df['status'] != 'FILLED'].get('hypothetical_pnl_per_share', []).dropna()) + list(filled_trades_for_hypo['pnl_per_share'].dropna())
        
        if hypo_pnl_list:
            winning_setups = [p for p in hypo_pnl_list if p > 0]
            losing_setups = [p for p in hypo_pnl_list if p <= 0]
            hypothetical_win_rate = (len(winning_setups) / len(hypo_pnl_list)) * 100 if hypo_pnl_list else 0
            gross_hypo_profit = sum(winning_setups)
            gross_hypo_loss = abs(sum(losing_setups))
            hypothetical_profit_factor = gross_hypo_profit / gross_hypo_loss if gross_hypo_loss > 0 else float('inf')

    params_str = "INPUT PARAMETERS:\n-----------------\n"
    for key, value in cfg.items(): params_str += f"{key.replace('_', ' ').title()}: {value}\n"
    summary_content = f"""BACKTEST SUMMARY REPORT (STRATEGY 1: FAST ENTRY FILTER CONFIRM)
===================================================================
{params_str}
REALISTIC PERFORMANCE (CAPITAL CONSTRAINED):
--------------------------------------------
Final Equity: {final_equity:,.2f}
Net P&L: {net_pnl:,.2f}
CAGR: {cagr:.2f}%
Max Drawdown: {max_drawdown:.1f}%
Total Core Trade Events (excl. Recon): {total_trades}
Win Rate (of core events): {win_rate:.1f}%
Profit Factor: {profit_factor:.2f}

HYPOTHETICAL PERFORMANCE (UNCONSTRAINED):
-----------------------------------------
Total Setups Identified: {len(all_setups_df)}
Strategy Win Rate (per setup): {hypothetical_win_rate:.1f}%
Strategy Profit Factor (per setup): {hypothetical_profit_factor:.2f}
"""
    
    # --- Save log files and print summary ---
    summary_filename = os.path.join(log_run_folder, f"{timestamp}_summary_report.txt")
    trades_filename = os.path.join(log_run_folder, f"{timestamp}_trades_detail.csv")
    all_setups_filename = os.path.join(log_run_folder, f"{timestamp}_all_setups_log.csv")
    
    with open(summary_filename, 'w') as f: f.write(summary_content)
    if not trades_df.empty:
        trades_df.to_csv(trades_filename, index=False)
    if not all_setups_df.empty:
        all_setups_df.to_csv(all_setups_filename, index=False)
    
    print(summary_content)
    print(f"Backtest completed in {time.time()-start_time:.2f} seconds")
    print(f"Reports saved to '{log_run_folder}'")


if __name__ == "__main__":
    run_backtest(config)
