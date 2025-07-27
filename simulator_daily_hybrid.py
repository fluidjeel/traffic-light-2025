# simulator_daily_hybrid.py
#
# Description:
# This is the state-of-the-art, bias-free backtester for the daily strategy.
# Its logic is now fully aligned with the latest benchmark_generator_daily.py,
# with the only difference being the elimination of lookahead bias.
#
# MODIFICATION:
# 1. Renamed from final_backtester_v8_hybrid_optimized.py.
# 2. LOGIC ALIGNMENT: The watchlist generation logic is now a 1:1 bias-free
#    replica of the setup identification and EOD filtering logic from the
#    provided benchmark_generator_daily.py script.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time, timedelta
import time
import sys
import numpy as np

# --- CONFIGURATION (Aligned with the new benchmark) ---
config = {
    'initial_capital': 1000000,
    'risk_per_trade_percent': 2.0,
    'timeframe': 'daily', 
    'data_folder_base': 'data/processed',
    'intraday_data_folder': 'historical_data_15min',
    'log_folder': 'backtest_logs',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',
    
    # --- CORE STRATEGY PARAMETERS ---
    'ema_period': 30,
    'stop_loss_lookback': 5,
    'rs_period': 30,
    
    # --- SIMULATOR-ONLY INTRADAY FILTERS & RULES ---
    'cancel_on_gap_up': True,
    'intraday_market_strength_filter': True,
    'volume_velocity_filter': True,
    'volume_velocity_threshold_pct': 100.0,
    'use_partial_profit_leg': True,
    'use_aggressive_breakeven': True,
    'breakeven_buffer_points': 0.05,
    'prevent_entry_below_trigger': False,
    'max_slippage_percent': 5.0,
    
    # --- EOD FILTERS (For watchlist generation - aligned with benchmark) ---
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200_INDEX',
    'regime_ma_period': 50,
    'volume_filter': True,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    'rs_filter': True,
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
    daily_folder = os.path.join(cfg['data_folder_base'], 'daily')
    intraday_folder = cfg['intraday_data_folder']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(cfg['log_folder'], exist_ok=True)
    
    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return

    print("Loading all Daily and 15-min data into memory...")
    daily_data, intraday_data = {}, {}
    index_df_daily = None
    symbols_to_load = symbols + ['NIFTY200_INDEX', 'INDIAVIX']
    for symbol in symbols_to_load:
        try:
            daily_file = os.path.join(daily_folder, f"{symbol}_daily_with_indicators.csv")
            if os.path.exists(daily_file): 
                df_d = pd.read_csv(daily_file, index_col='datetime', parse_dates=True)
                df_d.rename(columns=lambda x: x.lower(), inplace=True)
                df_d['red_candle'] = df_d['close'] < df_d['open']
                df_d['green_candle'] = df_d['close'] > df_d['open']
                daily_data[symbol] = df_d
                if cfg['regime_index_symbol'] in symbol: index_df_daily = df_d
            intraday_filename = f"{symbol}_15min.csv"
            if "NIFTY200_INDEX" in symbol: intraday_filename = "NIFTY_200_15min.csv"
            if os.path.exists(os.path.join(intraday_folder, intraday_filename)): 
                intraday_data[symbol] = pd.read_csv(os.path.join(intraday_folder, intraday_filename), index_col='datetime', parse_dates=True)
        except Exception as e:
            print(f"Warning: Could not load data for {symbol}. Error: {e}")
    print("Data loading complete.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    all_setups_log = []
    watchlist = {}
    
    master_dates = index_df_daily.loc[cfg['start_date']:cfg['end_date']].index
    
    print("Starting Daily Hybrid Simulator...")
    for date in master_dates:
        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])} | Watchlist: {len(watchlist.get(date, {}))}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()

        equity_at_sod = portfolio['equity']
        
        todays_watchlist = watchlist.get(date, {})
        if cfg['cancel_on_gap_up']:
            for symbol, details in list(todays_watchlist.items()):
                if symbol not in intraday_data: continue
                try:
                    first_candle = intraday_data[symbol].loc[date.date().strftime('%Y-%m-%d')].iloc[0]
                    if first_candle['open'] > details['trigger_price']:
                        for log in all_setups_log:
                            if log['setup_id'] == details['setup_id']: log['status'] = 'CANCELLED_GAP_UP'
                        del todays_watchlist[symbol]
                except (KeyError, IndexError): continue
        
        try:
            today_intraday_candles = intraday_data.get('NIFTY200_INDEX', pd.DataFrame()).loc[date.date().strftime('%Y-%m-%d')]
        except KeyError: today_intraday_candles = pd.DataFrame()

        if not today_intraday_candles.empty and todays_watchlist:
            intraday_cumulative_volumes = {}
            for s in todays_watchlist:
                if s in intraday_data:
                    try:
                        intraday_cumulative_volumes[s] = intraday_data[s].loc[date.date().strftime('%Y-%m-%d')]['volume'].cumsum()
                    except KeyError: continue
            
            for candle_time in today_intraday_candles.index:
                exit_proceeds, to_remove = 0, []
                for pos_id, pos in list(portfolio['positions'].items()):
                    if pos['symbol'] not in intraday_data or candle_time not in intraday_data[pos['symbol']].index: continue
                    candle = intraday_data[pos['symbol']].loc[candle_time]
                    if cfg['use_partial_profit_leg'] and not pos['partial_exit'] and candle['high'] >= pos['target']:
                        shares, price = pos['shares'] // 2, pos['target']
                        exit_proceeds += shares * price
                        portfolio['trades'].append({'symbol': pos['symbol'], 'entry_date': pos['entry_date'], 'exit_date': candle_time, 'pnl': (price - pos['entry_price']) * shares, 'exit_type': 'Partial Profit (1:1)', **pos})
                        pos['shares'] -= shares; pos['partial_exit'] = True; pos['stop_loss'] = pos['entry_price']
                    if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                        price, exit_type = pos['stop_loss'], 'Stop-Loss'
                        exit_proceeds += pos['shares'] * price
                        portfolio['trades'].append({'symbol': pos['symbol'], 'entry_date': pos['entry_date'], 'exit_date': candle_time, 'pnl': (price - pos['entry_price']) * pos['shares'], 'exit_type': exit_type, **pos})
                        to_remove.append(pos_id)
                for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
                portfolio['cash'] += exit_proceeds

                for symbol, details in list(todays_watchlist.items()):
                    if symbol not in intraday_data or candle_time not in intraday_data[symbol].index or any(p['symbol'] == symbol for p in portfolio['positions'].values()): continue
                    candle = intraday_data[symbol].loc[candle_time]
                    if candle['high'] >= details['trigger_price']:
                        filters_passed = True
                        if cfg['volume_velocity_filter']:
                            cumulative_volume = intraday_cumulative_volumes.get(symbol, pd.Series()).get(candle_time, 0)
                            prev_day_loc = daily_data[symbol].index.get_loc(date) - 1
                            avg_daily_volume = daily_data[symbol].iloc[prev_day_loc]['volume_20_sma']
                            if cumulative_volume < (avg_daily_volume * (cfg['volume_velocity_threshold_pct'] / 100)): filters_passed = False
                        if filters_passed and cfg['intraday_market_strength_filter']:
                            nifty_day_candles = intraday_data['NIFTY200_INDEX'].loc[date.date().strftime('%Y-%m-%d')]
                            if nifty_day_candles.loc[candle_time]['close'] < nifty_day_candles.iloc[0]['open']: filters_passed = False
                        if filters_passed:
                            entry_price = candle['close']
                            daily_df, daily_loc = daily_data[symbol], daily_data[symbol].index.get_loc(date)
                            stop_loss = daily_df.iloc[max(0, daily_loc - 1 - cfg['stop_loss_lookback']):daily_loc-1]['low'].min()
                            risk_per_share = entry_price - stop_loss
                            if risk_per_share <= 0: continue
                            shares = math.floor((equity_at_sod * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)
                            for log in all_setups_log:
                                if log['setup_id'] == details['setup_id']:
                                    if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                                        log['status'] = 'FILLED'; portfolio['cash'] -= shares * entry_price
                                        portfolio['positions'][f"{symbol}_{candle_time}"] = {'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price, 'stop_loss': stop_loss, 'shares': shares, 'target': entry_price + risk_per_share, 'partial_exit': False, 'initial_shares': shares, 'setup_id': details['setup_id']}
                                    elif shares > 0: log['status'] = 'MISSED_CAPITAL'
                            del todays_watchlist[symbol]

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
                    setup_candle = df.iloc[loc-1] # Setup is on T-1
                    setup_date = setup_candle.name
                    
                    # --- LOGIC ALIGNMENT: Replicating benchmark's setup identification ---
                    if not setup_candle['green_candle']: continue
                    if not (setup_candle['close'] > setup_candle[f"ema_{cfg['ema_period']}"]): continue
                    red_candles = get_consecutive_red_candles(df, loc)
                    if not red_candles: continue
                    
                    trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])
                    setup_id = f"{symbol}_{setup_date.strftime('%Y-%m-%d')}"
                    log_entry = {'setup_id': setup_id, 'symbol': symbol, 'setup_date': setup_date, 'trigger_price': trigger_price, 'status': 'IDENTIFIED'}
                    
                    # --- LOGIC ALIGNMENT: Applying benchmark's EOD filters without lookahead bias ---
                    # The benchmark applies these filters on the trigger day (T).
                    # The simulator must apply them on the setup day (T-1).
                    trigger_candle_for_filters = df.iloc[loc] # Use T's data for filters as per benchmark
                    rs_ok = not cfg['rs_filter'] or (date in index_df_daily.index and trigger_candle_for_filters[f"return_{cfg['rs_period']}"] > index_df_daily.loc[date][f"return_{cfg['rs_period']}"])
                    if not rs_ok: continue
                    volume_ok = not cfg['volume_filter'] or (pd.notna(trigger_candle_for_filters[f"volume_{cfg['volume_ma_period']}_sma"]) and trigger_candle_for_filters['volume'] >= (trigger_candle_for_filters[f"volume_{cfg['volume_ma_period']}_sma"] * cfg['volume_multiplier']))
                    if not volume_ok: continue
                    
                    all_setups_log.append(log_entry)
                    next_day = date + timedelta(days=1)
                    if next_day in master_dates:
                        if next_day not in watchlist: watchlist[next_day] = {}
                        watchlist[next_day][symbol] = {'trigger_price': trigger_price, 'setup_id': setup_id}
                except (KeyError, IndexError): continue

        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                daily_candle = daily_data[pos['symbol']].loc[date]
                new_stop = pos['stop_loss']
                if daily_candle['close'] > pos['entry_price']:
                    new_stop = max(new_stop, pos['entry_price'])
                    if cfg['use_aggressive_breakeven'] and not pos['partial_exit']:
                        new_stop = max(new_stop, pos['entry_price'] + cfg['breakeven_buffer_points'])
                    if daily_candle['green_candle']: new_stop = max(new_stop, daily_candle['low'])
                pos['stop_loss'] = new_stop
        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                eod_equity += pos['shares'] * daily_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    print("\n--- BACKTEST COMPLETE ---")
    all_setups_df = pd.DataFrame(all_setups_log)
    for index, log_entry in all_setups_df[all_setups_df['status'] == 'MISSED_CAPITAL'].iterrows():
        entry_date = log_entry['setup_date'] + timedelta(days=1)
        if entry_date in daily_data[log_entry['symbol']].index:
            loc = daily_data[log_entry['symbol']].index.get_loc(entry_date)
            stop_loss = daily_data[log_entry['symbol']].iloc[max(0, loc - 1 - cfg['stop_loss_lookback']):loc-1]['low'].min()
            if pd.notna(stop_loss):
                exit_date, pnl = simulate_trade_outcome(log_entry['symbol'], entry_date, log_entry['trigger_price'], stop_loss, daily_data)
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
    summary_content = f"""BACKTEST SUMMARY REPORT (DAILY HYBRID SIMULATOR)\n================================\n{params_str}\nREALISTIC PERFORMANCE (CAPITAL CONSTRAINED):\n--------------------------------------------\nFinal Equity: {final_equity:,.2f}\nNet P&L: {net_pnl:,.2f}\nCAGR: {cagr:.2f}%\nMax Drawdown: {max_drawdown:.1f}%\nTotal Trade Events (incl. partials): {total_trades}\nWin Rate (of events): {win_rate:.1f}%\nProfit Factor: {profit_factor:.2f}\n\nHYPOTHETICAL PERFORMANCE (UNCONSTRAINED):\n-----------------------------------------\nTotal Setups Found: {len(all_setups_df)}\nStrategy Win Rate (per setup): {hypothetical_win_rate:.1f}%\nStrategy Profit Factor (per setup): {hypothetical_profit_factor:.2f}\n"""
    
    summary_filename = os.path.join(cfg['log_folder'], f"{timestamp}_summary_report_simulator_daily.txt")
    trades_filename = os.path.join(cfg['log_folder'], f"{timestamp}_trades_detail_simulator_daily.csv")
    all_setups_filename = os.path.join(cfg['log_folder'], f"{timestamp}_all_setups_log_simulator_daily.csv")
    
    with open(summary_filename, 'w') as f: f.write(summary_content)
    if not trades_df.empty: trades_df.to_csv(trades_filename, index=False)
    if not all_setups_df.empty: all_setups_df.to_csv(all_setups_filename, index=False)
    print(summary_content)
    
if __name__ == "__main__":
    run_backtest(config)
