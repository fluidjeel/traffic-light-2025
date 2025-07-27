# final_backtester_v8_hybrid_optimized.py
#
# Description:
# This is the state-of-the-art, bias-free backtester for the daily strategy.
# It identifies setups on completed T-1 data and executes on Day T using
# real-time intraday data and a suite of advanced, toggleable features.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time
import time
import sys
import numpy as np

# --- V8 HYBRID CONFIGURATION ---
config = {
    'initial_capital': 1000000,
    'risk_per_trade_percent': 4.0,
    'timeframe': 'daily',
    'data_folder_base': 'data/processed',
    'intraday_data_folder': 'historical_data_15min',
    'log_folder': 'backtest_logs',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',

    # --- CORE STRATEGY PARAMETERS (DAILY) ---
    'ema_period': 30,
    'stop_loss_lookback': 5, # In days
    'rs_period': 30,

    # --- REAL-TIME FILTERS (INTRADAY) ---
    'intraday_market_strength_filter': True,
    'intraday_rs_filter': True,
    'volume_velocity_filter': True,
    'volume_velocity_threshold_pct': 100.0,

    # --- ENTRY & TRADE MANAGEMENT ---
    'max_slippage_percent': 5.0,
    'use_dynamic_slippage': True,
    'cancel_on_gap_up': True,
    'prevent_entry_below_trigger': True,
    'use_aggressive_breakeven': True,
    'breakeven_buffer_points': 0.05,
    'use_partial_profit_leg': True,

    # --- MISC ---
    'log_missed_trades': True
}

def calculate_dynamic_slippage(avg_daily_volume, vix_value):
    base_slippage = 0.0015
    safe_volume = max(avg_daily_volume, 10000)
    liquidity_adjustment_factor = 0.001
    liquidity_adjustment = (1000000 / safe_volume) * liquidity_adjustment_factor
    liquidity_adjustment = min(liquidity_adjustment, 0.01)
    volatility_adjustment = (vix_value / 15.0) * 0.0005
    total_slippage_pct = base_slippage + liquidity_adjustment + volatility_adjustment
    return total_slippage_pct

def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 2
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def simulate_trade_outcome(symbol, entry_date, entry_price, stop_loss, daily_data, cfg):
    df = daily_data[symbol]
    target_price = entry_price + (entry_price - stop_loss)
    current_stop = stop_loss
    partial_exit_pnl = 0
    final_pnl = 0
    leg1_sold = False
    exit_date = None
    aggressive_be_triggered = False
    trade_dates = df.loc[entry_date:].index[1:]
    for date in trade_dates:
        if date not in df.index: continue
        candle = df.loc[date]
        if cfg['use_partial_profit_leg'] and not leg1_sold and candle['high'] >= target_price:
            partial_exit_pnl = target_price - entry_price
            leg1_sold = True
            current_stop = entry_price
        if candle['low'] <= current_stop:
            final_pnl = current_stop - entry_price
            exit_date = date
            break
        if candle['close'] > entry_price:
            new_stop = current_stop
            new_stop = max(new_stop, entry_price)
            if cfg['use_aggressive_breakeven'] and not leg1_sold and not aggressive_be_triggered:
                breakeven_stop = entry_price + cfg['breakeven_buffer_points']
                new_stop = max(new_stop, breakeven_stop)
                aggressive_be_triggered = True
            if candle['green_candle']:
                new_stop = max(new_stop, candle['low'])
            current_stop = new_stop
    if exit_date is None:
        exit_date = df.index[-1]
        final_pnl = df.iloc[-1]['close'] - entry_price
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
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}")
        return

    print("Loading all Daily and 15-min data into memory...")
    daily_data, intraday_data = {}, {}
    symbols_to_load = symbols + ['NIFTY200_INDEX', 'INDIAVIX']
    for symbol in symbols_to_load:
        try:
            daily_file = os.path.join(daily_folder, f"{symbol}_daily_with_indicators.csv")
            if "NIFTY200_INDEX" in symbol:
                intraday_filename = "NIFTY_200_15min.csv"
            elif "INDIAVIX" in symbol:
                intraday_filename = None
            else:
                intraday_filename = f"{symbol}_15min.csv"

            if os.path.exists(daily_file):
                df_d = pd.read_csv(daily_file, index_col='datetime', parse_dates=True)
                df_d['red_candle'] = df_d['close'] < df_d['open']
                df_d['green_candle'] = df_d['close'] > df_d['open']
                daily_data[symbol] = df_d
            if intraday_filename and os.path.exists(os.path.join(intraday_folder, intraday_filename)):
                intraday_data[symbol] = pd.read_csv(os.path.join(intraday_folder, intraday_filename), index_col='datetime', parse_dates=True)
        except Exception as e:
            print(f"Warning: Could not load all data for {symbol}. Error: {e}")
    print("Data loading complete.")
    vix_available = 'INDIAVIX' in daily_data
    if cfg['use_dynamic_slippage'] and not vix_available:
        print("Warning: Dynamic slippage enabled but INDIAVIX data not found. Reverting to simple fill logic.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    all_setups_log = []
    master_dates = daily_data.get('NIFTY200_INDEX', pd.DataFrame()).loc[cfg['start_date']:cfg['end_date']].index
    print("Starting V8 Hybrid Optimized backtest simulation...")
    for date in master_dates:
        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()
        watchlist = {}
        equity_at_sod = portfolio['equity']
        for symbol in symbols:
            if symbol not in daily_data: continue
            df_daily = daily_data[symbol]
            try:
                loc = df_daily.index.get_loc(date)
                if loc < 2: continue
                prev1 = df_daily.iloc[loc-1]
                if not prev1['green_candle']: continue
                if not (prev1['close'] > prev1[f"ema_{cfg['ema_period']}"]): continue
                red_candles = get_consecutive_red_candles(df_daily, loc)
                if not red_candles: continue
                trigger_price = max([c['high'] for c in red_candles] + [prev1['high']])
                watchlist[symbol] = {'trigger_price': trigger_price, 'status': 'monitoring'}
            except (KeyError, IndexError):
                continue
        if cfg['cancel_on_gap_up']:
            symbols_to_remove = []
            for symbol, details in watchlist.items():
                if symbol not in intraday_data: continue
                try:
                    first_candle = intraday_data[symbol].loc[date.date().strftime('%Y-%m-%d')].iloc[0]
                    if first_candle['open'] > details['trigger_price']:
                        symbols_to_remove.append(symbol)
                        log_entry = {'symbol': symbol, 'setup_date': date, 'trigger_price': details['trigger_price'], 'status': 'CANCELLED_GAP_UP'}
                        all_setups_log.append(log_entry)
                except (KeyError, IndexError):
                    continue
            for symbol in symbols_to_remove:
                del watchlist[symbol]
        try:
            today_intraday_candles = intraday_data.get('NIFTY200_INDEX', pd.DataFrame()).loc[date.date().strftime('%Y-%m-%d')]
        except KeyError:
            today_intraday_candles = pd.DataFrame()
        if not today_intraday_candles.empty and watchlist:
            intraday_cumulative_volumes = {}
            for symbol in watchlist:
                if symbol in intraday_data:
                    try:
                        day_candles = intraday_data[symbol].loc[date.date().strftime('%Y-%m-%d')]
                        if not day_candles.empty:
                            intraday_cumulative_volumes[symbol] = day_candles['volume'].cumsum()
                    except KeyError:
                        continue
            for candle_time in today_intraday_candles.index:
                exit_proceeds = 0; to_remove = []
                for pos_id, pos in list(portfolio['positions'].items()):
                    if pos['symbol'] not in intraday_data or candle_time not in intraday_data[pos['symbol']].index: continue
                    candle = intraday_data[pos['symbol']].loc[candle_time]
                    if cfg['use_partial_profit_leg']:
                        if not pos['partial_exit'] and candle['high'] >= pos['target']:
                            shares = pos['shares'] // 2; price = pos['target']; exit_proceeds += shares * price
                            portfolio['trades'].append({'symbol': pos['symbol'], 'entry_date': pos['entry_date'], 'exit_date': candle_time, 'pnl': (price - pos['entry_price']) * shares, 'exit_type': 'Partial Profit (1:1)', **pos})
                            pos['shares'] -= shares; pos['partial_exit'] = True; pos['stop_loss'] = pos['entry_price']
                    if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                        price = pos['stop_loss']
                        exit_type = 'Stop-Loss'
                        if cfg['use_aggressive_breakeven']:
                            breakeven_stop_level = pos.get('breakeven_stop_level', -1)
                            if breakeven_stop_level > 0 and price == breakeven_stop_level:
                                exit_type = 'Aggressive Breakeven Stop'
                        exit_proceeds += pos['shares'] * price
                        portfolio['trades'].append({'symbol': pos['symbol'], 'entry_date': pos['entry_date'], 'exit_date': candle_time, 'pnl': (price - pos['entry_price']) * pos['shares'], 'exit_type': exit_type, **pos})
                        to_remove.append(pos_id)
                for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
                portfolio['cash'] += exit_proceeds
                for symbol, details in list(watchlist.items()):
                    if symbol not in intraday_data or candle_time not in intraday_data[symbol].index: continue
                    if any(p['symbol'] == symbol for p in portfolio['positions'].values()): continue
                    candle = intraday_data[symbol].loc[candle_time]
                    status = details['status']
                    if status == 'monitoring' and candle['high'] >= details['trigger_price']:
                        details['status'] = 'triggered'
                    if details['status'] == 'triggered':
                        if candle['close'] > details['trigger_price'] * (1 + cfg['max_slippage_percent'] / 100):
                            del watchlist[symbol]
                            continue
                        filters_passed = True
                        if cfg['volume_velocity_filter']:
                            cumulative_volume = intraday_cumulative_volumes.get(symbol, pd.Series()).get(candle_time, 0)
                            prev_day_loc = daily_data[symbol].index.get_loc(date) - 1
                            avg_daily_volume = daily_data[symbol].iloc[prev_day_loc]['volume_20_sma']
                            if cumulative_volume < (avg_daily_volume * (cfg['volume_velocity_threshold_pct'] / 100)):
                                filters_passed = False
                        if filters_passed and cfg['intraday_market_strength_filter']:
                            nifty_day_candles = intraday_data['NIFTY200_INDEX'].loc[date.date().strftime('%Y-%m-%d')]
                            if nifty_day_candles.loc[candle_time]['close'] < nifty_day_candles.iloc[0]['open']:
                                filters_passed = False
                        if filters_passed and cfg['prevent_entry_below_trigger']:
                            if candle['close'] < details['trigger_price']:
                                filters_passed = False
                                log_entry = {'symbol': symbol, 'setup_date': date, 'trigger_price': details['trigger_price'], 'status': 'CANCELLED_FAILED_BREAKOUT'}
                                all_setups_log.append(log_entry)
                                del watchlist[symbol]
                        if filters_passed:
                            if cfg['use_dynamic_slippage'] and vix_available:
                                daily_loc = daily_data[symbol].index.get_loc(date)
                                prev_day_loc = daily_loc - 1
                                avg_vol = daily_data[symbol].iloc[prev_day_loc]['volume_20_sma']
                                vix_value = daily_data['INDIAVIX'].loc[date]['close'] if date in daily_data['INDIAVIX'].index else 15.0
                                slippage_pct = calculate_dynamic_slippage(avg_vol, vix_value)
                                entry_price = candle['close'] * (1 + slippage_pct)
                            else:
                                entry_price = candle['close']
                            daily_df = daily_data[symbol]
                            daily_loc = daily_df.index.get_loc(date)
                            stop_loss = daily_df.iloc[max(0, daily_loc - 1 - cfg['stop_loss_lookback']):daily_loc]['low'].min()
                            if pd.isna(stop_loss):
                                stop_loss = daily_df.iloc[daily_loc - 1]['low']
                            risk_per_share = entry_price - stop_loss
                            if risk_per_share <= 0: continue
                            shares = math.floor((equity_at_sod * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)
                            prev_day_candle = daily_data[symbol].iloc[daily_loc - 1]
                            if prev_day_candle.name not in daily_data['NIFTY200_INDEX'].index:
                                continue
                            prev_day_index = daily_data['NIFTY200_INDEX'].loc[prev_day_candle.name]
                            return_col_name = f"return_{cfg['rs_period']}"
                            log_entry = {'symbol': symbol, 'setup_date': date, 'trigger_price': details['trigger_price'], 'status': '', 'stock_rs_minus_nifty_rs': prev_day_candle.get(return_col_name, 0) - prev_day_index.get(return_col_name, 0), 'volume_vs_avg_multiplier': prev_day_candle['volume'] / prev_day_candle['volume_20_sma'] if prev_day_candle['volume_20_sma'] > 0 else 0}
                            if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                                log_entry['status'] = 'FILLED'
                                portfolio['cash'] -= shares * entry_price
                                portfolio['positions'][f"{symbol}_{candle_time}"] = {'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price, 'stop_loss': stop_loss, 'shares': shares, 'target': entry_price + risk_per_share, 'partial_exit': False, 'portfolio_equity_on_entry': equity_at_sod, 'risk_per_share': risk_per_share, 'initial_shares': shares, 'initial_stop_loss': stop_loss, 'is_aggressively_managed': False}
                            elif shares > 0:
                                log_entry['status'] = 'MISSED_CAPITAL'
                            all_setups_log.append(log_entry)
                            del watchlist[symbol]
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                daily_candle = daily_data[pos['symbol']].loc[date]
                new_stop = pos['stop_loss']
                if daily_candle['close'] > pos['entry_price']:
                    new_stop = max(new_stop, pos['entry_price'])
                    if cfg['use_aggressive_breakeven'] and not pos['partial_exit'] and not pos['is_aggressively_managed']:
                        breakeven_stop = pos['entry_price'] + cfg['breakeven_buffer_points']
                        pos['breakeven_stop_level'] = breakeven_stop
                        new_stop = max(new_stop, breakeven_stop)
                        pos['is_aggressively_managed'] = True
                    if daily_candle['green_candle']:
                        new_stop = max(new_stop, daily_candle['low'])
                pos['stop_loss'] = new_stop
        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                eod_equity += pos['shares'] * daily_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})
    print("\n--- BACKTEST COMPLETE ---")
    for log_entry in all_setups_log:
        if log_entry.get('status') in ['CANCELLED_GAP_UP', 'CANCELLED_FAILED_BREAKOUT']: continue
        if 'symbol' not in log_entry or 'setup_date' not in log_entry: continue
        daily_df = daily_data[log_entry['symbol']]
        loc = daily_df.index.get_loc(log_entry['setup_date'])
        stop_loss = daily_df.iloc[max(0, loc - 1 - cfg['stop_loss_lookback']):loc]['low'].min()
        if pd.isna(stop_loss):
            stop_loss = daily_df.iloc[loc - 1]['low']
        exit_date, pnl = simulate_trade_outcome(log_entry['symbol'], log_entry['setup_date'], log_entry['trigger_price'], stop_loss, daily_data, cfg)
        log_entry['hypothetical_exit_date'] = exit_date
        log_entry['hypothetical_pnl_per_share'] = pnl
    final_equity = portfolio['equity']
    net_pnl = final_equity - config['initial_capital']
    equity_df = pd.DataFrame(portfolio['daily_values'])
    equity_df.set_index('date', inplace=True)
    if not equity_df.empty:
        years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = ((final_equity / config['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
        peak = equity_df['equity'].cummax(); drawdown = (equity_df['equity'] - peak) / peak; max_drawdown = abs(drawdown.min()) * 100
    else: cagr, max_drawdown = 0, 0
    trades_df = pd.DataFrame(portfolio['trades'])
    total_trades, win_rate, profit_factor, avg_win, avg_loss = 0, 0, 0, 0, 0
    if not trades_df.empty:
        winning_trades = trades_df[trades_df['pnl'] > 0]; losing_trades = trades_df[trades_df['pnl'] <= 0]
        total_trades = len(trades_df)
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        gross_profit = winning_trades['pnl'].sum(); gross_loss = abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
    all_setups_df = pd.DataFrame(all_setups_log)
    hypothetical_win_rate, hypothetical_profit_factor = 0, 0
    if not all_setups_df.empty:
        valid_setups_df = all_setups_df[~all_setups_df['status'].isin(['CANCELLED_GAP_UP', 'CANCELLED_FAILED_BREAKOUT'])]
        filled_trades_for_hypo = trades_df[trades_df['exit_type'] != 'Partial Profit (1:1)'].copy()
        if 'initial_shares' in filled_trades_for_hypo.columns and not filled_trades_for_hypo.empty:
            filled_trades_for_hypo = filled_trades_for_hypo[filled_trades_for_hypo['initial_shares'] > 0]
            filled_trades_for_hypo['pnl_per_share'] = filled_trades_for_hypo['pnl'] / filled_trades_for_hypo['initial_shares']
        else:
            filled_trades_for_hypo['pnl_per_share'] = 0
        hypo_pnl_list = list(valid_setups_df[valid_setups_df['status'] == 'MISSED_CAPITAL'].get('hypothetical_pnl_per_share', [])) + list(filled_trades_for_hypo['pnl_per_share'])
        if hypo_pnl_list:
            winning_setups = [p for p in hypo_pnl_list if p > 0]
            losing_setups = [p for p in hypo_pnl_list if p <= 0]
            hypothetical_win_rate = (len(winning_setups) / len(hypo_pnl_list)) * 100 if hypo_pnl_list else 0
            gross_hypo_profit = sum(winning_setups)
            gross_hypo_loss = abs(sum(losing_setups))
            hypothetical_profit_factor = gross_hypo_profit / gross_hypo_loss if gross_hypo_loss > 0 else float('inf')
    params_str = "INPUT PARAMETERS:\n-----------------\n"
    for key, value in cfg.items():
        key_formatted = key.replace('_', ' ').title()
        params_str += f"{key_formatted}: {value}\n"
    summary_content = f"""BACKTEST SUMMARY REPORT (V8 HYBRID - OPTIMIZED)
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
Total Setups Found: {len(all_setups_log)}
Strategy Win Rate (per setup): {hypothetical_win_rate:.1f}%
Strategy Profit Factor (per setup): {hypothetical_profit_factor:.2f}
"""
    summary_filename = os.path.join(cfg['log_folder'], f"{timestamp}_summary_report_v8_optimized.txt")
    trades_filename = os.path.join(cfg['log_folder'], f"{timestamp}_trades_detail_v8_optimized.csv")
    all_setups_filename = os.path.join(cfg['log_folder'], f"{timestamp}_all_setups_log_v8_optimized.csv")
    with open(summary_filename, 'w') as f: f.write(summary_content)
    if not trades_df.empty:
        trades_df.to_csv(trades_filename, index=False)
    if all_setups_log:
        all_setups_df = pd.DataFrame(all_setups_log)
        all_setups_df.to_csv(all_setups_filename, index=False)
    print(summary_content)
    print(f"Backtest completed in {time.time()-start_time:.2f} seconds")
    print(f"Reports saved to '{cfg['log_folder']}'")

if __name__ == "__main__":
    run_backtest(config)
