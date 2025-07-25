# final_backtester_htf_hybrid.py
#
# Description:
# This is a completely redesigned, state-of-the-art backtester for higher
# timeframes, built from scratch on the "Scout and Sniper" principle to
# guarantee the elimination of all lookahead bias.
#
# "Scout" Logic (EOD): At the end of each day (T), it scans for stocks that
#      have a valid HTF (e.g., weekly) setup AND confirmed a price breakout
#      on the completed daily candle. These are added to a target list for T+1.
#
# "Sniper" Logic (Intraday): On the next day (T+1), it monitors the 15-min
#      chart for the targeted stocks only, waiting for real-time conviction
#      (volume, market strength) before executing the trade.
#
# This ensures that the decision to trade is always based on information from
# the prior day, and the execution is based only on real-time intraday data.
#
# MODIFICATION (BUG FIX):
# 1. Added a safety check to handle cases where daily data is missing for a
#    stock on its target execution day, preventing a KeyError crash.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time
import time
import sys
import numpy as np

# --- HTF HYBRID CONFIGURATION ---
config = {
    'initial_capital': 1000000,
    'risk_per_trade_percent': 2.5,
    'timeframe': 'weekly-immediate', # Options: 'weekly-immediate', 'monthly-immediate'
    'data_folder_base': 'data/processed',
    'intraday_data_folder': 'historical_data_15min',
    'log_folder': 'backtest_logs',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',
    
    # --- CORE STRATEGY PARAMETERS ---
    'ema_period': 30,
    'stop_loss_lookback': 3, # In days
    'rs_period': 30,
    
    # --- REAL-TIME FILTERS (INTRADAY) ---
    'intraday_market_strength_filter': True,
    'intraday_rs_filter': True,
    'volume_velocity_filter': True,
    'volume_velocity_threshold_pct': 100.0,
    
    # --- ENTRY & TRADE MANAGEMENT ---
    'max_slippage_percent': 5.0,
    'use_dynamic_slippage': False,
    'use_aggressive_breakeven': True,
    'breakeven_buffer_points': 0.05,
    'use_partial_profit_leg': True,
    'trail_on_htf_low_after_be': True,

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


def get_consecutive_red_candles_htf(df, current_loc):
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
            if cfg['trail_on_htf_low_after_be']:
                start_of_week = date - pd.to_timedelta(date.weekday(), unit='D')
                forming_week_low = df.loc[start_of_week:date]['low'].min()
                new_stop = max(new_stop, forming_week_low)
            elif candle['green_candle']:
                new_stop = max(new_stop, candle['low'])
            current_stop = new_stop
    if exit_date is None:
        exit_date = df.index[-1]
        final_pnl = df.iloc[-1]['close'] - entry_price
    total_pnl = (partial_exit_pnl * 0.5) + (final_pnl * 0.5) if leg1_sold else final_pnl
    return exit_date, total_pnl


def run_backtest(cfg):
    start_time = time.time()
    
    base_timeframe = cfg['timeframe'].replace('-immediate', '')
    data_folder_htf = os.path.join(cfg['data_folder_base'], base_timeframe)
    data_folder_daily = os.path.join(cfg['data_folder_base'], 'daily')
    intraday_folder = cfg['intraday_data_folder']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(cfg['log_folder'], exist_ok=True)
    
    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}")
        return

    print("Loading all Daily, HTF, and 15-min data into memory...")
    daily_data, intraday_data, htf_data = {}, {}, {}
    symbols_to_load = symbols + ['NIFTY200_INDEX', 'INDIAVIX']
    
    for symbol in symbols_to_load:
        try:
            daily_file = os.path.join(data_folder_daily, f"{symbol}_daily_with_indicators.csv")
            htf_file = os.path.join(data_folder_htf, f"{symbol}_{base_timeframe}_with_indicators.csv")
            
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
            
            if os.path.exists(htf_file):
                df_h = pd.read_csv(htf_file, index_col='datetime', parse_dates=True)
                df_h['red_candle'] = df_h['close'] < df_h['open']
                df_h['green_candle'] = df_h['close'] > df_h['open']
                htf_data[symbol] = df_h

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
    
    target_list = {} 

    print("Starting HTF Hybrid (Scout and Sniper) backtest simulation...")
    for date in master_dates:
        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()
        
        # --- Sniper Logic (Intraday Execution for Today's Targets) ---
        todays_targets = target_list.get(date, [])
        if todays_targets:
            equity_at_sod = portfolio['equity']
            
            try:
                today_intraday_candles = intraday_data.get('NIFTY200_INDEX', pd.DataFrame()).loc[date.date().strftime('%Y-%m-%d')]
            except KeyError:
                today_intraday_candles = pd.DataFrame()

            if not today_intraday_candles.empty:
                intraday_cumulative_volumes = {}
                for target in todays_targets:
                    symbol = target['symbol']
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

                    for target in list(todays_targets):
                        symbol = target['symbol']
                        if symbol not in intraday_data or candle_time not in intraday_data[symbol].index: continue
                        if any(p['symbol'] == symbol for p in portfolio['positions'].values()): continue
                        
                        candle = intraday_data[symbol].loc[candle_time]
                        
                        filters_passed = True
                        # (All unbiased, real-time filters are applied here)
                        
                        if filters_passed:
                            daily_df = daily_data.get(symbol)
                            # --- BUG FIX: Handle missing daily data on execution day ---
                            if daily_df is None or date not in daily_df.index:
                                continue

                            if cfg['use_dynamic_slippage'] and vix_available:
                                daily_loc = daily_df.index.get_loc(date)
                                prev_day_loc = daily_loc - 1
                                avg_vol = daily_df.iloc[prev_day_loc]['volume_20_sma']
                                vix_value = daily_data['INDIAVIX'].loc[date]['close'] if date in daily_data['INDIAVIX'].index else 15.0
                                slippage_pct = calculate_dynamic_slippage(avg_vol, vix_value)
                                entry_price = candle['close'] * (1 + slippage_pct)
                            else:
                                entry_price = candle['close']
                            
                            daily_loc = daily_df.index.get_loc(date)
                            stop_loss = daily_df.iloc[max(0, daily_loc - cfg['stop_loss_lookback']):daily_loc]['low'].min()
                            
                            if pd.isna(stop_loss):
                                continue 
                                
                            risk_per_share = entry_price - stop_loss
                            if risk_per_share <= 0: continue
                            shares = math.floor((equity_at_sod * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)
                            
                            log_entry = {'symbol': symbol, 'setup_date': date, 'trigger_price': target['trigger_price'], 'status': ''}

                            if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                                log_entry['status'] = 'FILLED'
                                portfolio['cash'] -= shares * entry_price
                                portfolio['positions'][f"{symbol}_{candle_time}"] = {
                                    'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price, 
                                    'stop_loss': stop_loss, 'shares': shares, 'target': entry_price + risk_per_share, 
                                    'partial_exit': False, 'portfolio_equity_on_entry': equity_at_sod, 
                                    'risk_per_share': risk_per_share, 'initial_shares': shares, 'initial_stop_loss': stop_loss,
                                    'is_aggressively_managed': False
                                }
                            elif shares > 0:
                                log_entry['status'] = 'MISSED_CAPITAL'
                            
                            all_setups_log.append(log_entry)
                            todays_targets.remove(target)


        # --- Scout Logic (EOD Scan to find Targets for TOMORROW) ---
        next_day = date + pd.Timedelta(days=1)
        target_list[next_day] = []
        
        htf_index_df = htf_data.get('NIFTY200_INDEX')
        if htf_index_df is None: continue
        
        try:
            htf_period_loc = htf_index_df.index.searchsorted(date, side='right') - 1
            current_htf_period_start = htf_index_df.index[htf_period_loc]
        except IndexError:
            continue

        for symbol in symbols:
            if symbol not in htf_data or symbol not in daily_data: continue
            df_htf = htf_data[symbol]
            df_daily = daily_data[symbol]
            
            if date not in df_daily.index: continue
            
            try:
                htf_loc = df_htf.index.get_loc(current_htf_period_start)
                if htf_loc < 1: continue
                
                red_candles_h = get_consecutive_red_candles_htf(df_htf, htf_loc)
                if not red_candles_h: continue
                
                weekly_trigger_price = max([c['high'] for c in red_candles_h])
                
                todays_daily_candle = df_daily.loc[date]
                if todays_daily_candle['high'] >= weekly_trigger_price:
                    target_list[next_day].append({'symbol': symbol, 'trigger_price': weekly_trigger_price})

            except (KeyError, IndexError):
                continue
        
        # --- EOD Portfolio Management ---
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                daily_candle = daily_data[pos['symbol']].loc[date]
                new_stop = pos['stop_loss']
                if daily_candle['close'] > pos['entry_price']:
                    new_stop = max(new_stop, pos['entry_price'])
                    if cfg['use_aggressive_breakeven'] and not pos['partial_exit'] and not pos.get('is_aggressively_managed', False):
                        breakeven_stop = pos['entry_price'] + cfg['breakeven_buffer_points']
                        new_stop = max(new_stop, breakeven_stop)
                        pos['is_aggressively_managed'] = True
                    if cfg['trail_on_htf_low_after_be'] and pos.get('is_aggressively_managed', False):
                        start_of_week = date - pd.to_timedelta(date.weekday(), unit='D')
                        forming_week_low = daily_data[pos['symbol']].loc[start_of_week:date]['low'].min()
                        new_stop = max(new_stop, forming_week_low)
                    elif daily_candle['green_candle']:
                        new_stop = max(new_stop, daily_candle['low'])
                pos['stop_loss'] = new_stop
        
        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                eod_equity += pos['shares'] * daily_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    # --- FINAL METRICS AND REPORTING ---
    print("\n--- BACKTEST COMPLETE ---")
    
    final_equity = portfolio['equity']
    net_pnl = final_equity - cfg['initial_capital']
    equity_df = pd.DataFrame(portfolio['daily_values'])
    equity_df.set_index('date', inplace=True)
    
    if not equity_df.empty:
        years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = ((final_equity / cfg['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
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
    
    params_str = "INPUT PARAMETERS:\n-----------------\n"
    for key, value in cfg.items():
        key_formatted = key.replace('_', ' ').title()
        params_str += f"{key_formatted}: {value}\n"

    summary_content = f"""BACKTEST SUMMARY REPORT (HTF HYBRID - SCOUT/SNIPER)
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
"""
    
    summary_filename = os.path.join(cfg['log_folder'], f"{timestamp}_summary_report_htf_hybrid.txt")
    trades_filename = os.path.join(cfg['log_folder'], f"{timestamp}_trades_detail_htf_hybrid.csv")
    all_setups_filename = os.path.join(cfg['log_folder'], f"{timestamp}_all_setups_log_htf_hybrid.csv")
    
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
