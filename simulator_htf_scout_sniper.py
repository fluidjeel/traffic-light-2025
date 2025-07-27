# final_backtester_htf_hybrid.py
#
# Description:
# This is a state-of-the-art, bias-free backtester for running the "immediate"
# entry strategy on higher timeframes (e.g., weekly, monthly).
#
# MODIFICATION 7 (LOGGING ENHANCEMENT): Added a unique `setup_id`.
# MODIFICATION 8 (BUG FIX): Corrected IndentationError.
# MODIFICATION 9 (BUG FIX): Restored missing EOD equity calculation.
# MODIFICATION 10 (BUG FIX): Added a check to prevent KeyError for missing dates.
# MODIFICATION 11 (BUG FIX): Corrected NameError and KeyError in final reporting.
# MODIFICATION 12 (BUG FIX): Corrected the core logic to properly implement the
#   "Scout and Sniper" architecture as per the project documentation.
# MODIFICATION 13 (BUG FIX): Restored the complete final reporting and logging section.
# MODIFICATION 14 (BUG FIX): Added the missing EOD Volume and RS filters to the
#   "Scout" logic to ensure it identifies the same setup universe as the benchmark.
# MODIFICATION 15 (ENHANCEMENT): Restored full Sniper execution logic and final
#   reporting block to match the project's standard level of detail.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time
import time
import sys
import numpy as np

# --- UNIFIED HTF CONFIGURATION ---
config = {
    'initial_capital': 1000000,
    'risk_per_trade_percent': 2.0,
    'timeframe': 'weekly-immediate', # Options: 'weekly-immediate', 'monthly-immediate'
    'data_folder_base': 'data/processed',
    'intraday_data_folder': 'historical_data_15min',
    'log_folder': 'backtest_logs',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',
    
    # --- CORE STRATEGY PARAMETERS ---
    'ema_period': 20,
    'stop_loss_lookback': 5, # In days
    'rs_period': 30,
    
    # --- REAL-TIME FILTERS (FOR HYBRID MODEL) ---
    'intraday_market_strength_filter': True,
    'intraday_rs_filter': True,
    'volume_velocity_filter': True,
    'volume_velocity_threshold_pct': 100.0,
    
    # --- ENTRY & TRADE MANAGEMENT (FOR HYBRID MODEL) ---
    'max_slippage_percent': 5.0,
    'use_dynamic_slippage': False,
    'cancel_on_gap_up': True,
    'prevent_entry_below_trigger': True,
    'use_aggressive_breakeven': False,
    'breakeven_buffer_points': 0.05,
    'use_partial_profit_leg': True,
    'trail_on_htf_low_after_be': True,

    # --- IMMINENCE FILTER CONFIG (FOR HYBRID MODEL) ---
    'use_imminence_filter': False,

    # --- BENCHMARK-ONLY FILTERS (NOW USED BY SCOUT) ---
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200_INDEX',
    'regime_ma_period': 50,
    'volume_filter': True,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    'rs_filter': True,
    
    # --- MISC ---
    'log_missed_trades': True
}

# --- Helper functions (unchanged) ---
def calculate_dynamic_slippage(avg_daily_volume, vix_value):
    base_slippage = 0.0015; safe_volume = max(avg_daily_volume, 10000); liquidity_adjustment_factor = 0.001
    liquidity_adjustment = (1000000 / safe_volume) * liquidity_adjustment_factor; liquidity_adjustment = min(liquidity_adjustment, 0.01)
    volatility_adjustment = (vix_value / 15.0) * 0.0005; total_slippage_pct = base_slippage + liquidity_adjustment + volatility_adjustment
    return total_slippage_pct
def get_consecutive_red_candles_htf(df, current_loc):
    red_candles = []; i = current_loc - 1
    while i >= 0 and df.iloc[i]['red_candle']: red_candles.append(df.iloc[i]); i -= 1
    return red_candles
def is_inside_day(daily_df, date):
    try:
        loc = daily_df.index.get_loc(date);
        if loc < 1: return False
        current_day, prior_day = daily_df.iloc[loc], daily_df.iloc[loc - 1]
        return current_day['high'] < prior_day['high'] and current_day['low'] > prior_day['low']
    except (KeyError, IndexError): return False
def is_nr7(daily_df, date, period=7):
    try:
        loc = daily_df.index.get_loc(date)
        if loc < period - 1: return False
        recent_ranges = (daily_df['high'] - daily_df['low']).iloc[loc - period + 1 : loc + 1]
        return recent_ranges.iloc[-1] == recent_ranges.min()
    except (KeyError, IndexError): return False
def simulate_trade_outcome(symbol, entry_date, entry_price, stop_loss, daily_data, cfg):
    df = daily_data[symbol]; target_price = entry_price + (entry_price - stop_loss); current_stop = stop_loss
    partial_exit_pnl, final_pnl, leg1_sold, exit_date, aggressive_be_triggered = 0, 0, False, None, False
    trade_dates = df.loc[entry_date:].index[1:]
    for date in trade_dates:
        if date not in df.index: continue
        candle = df.loc[date]
        if cfg['use_partial_profit_leg'] and not leg1_sold and candle['high'] >= target_price:
            partial_exit_pnl = target_price - entry_price; leg1_sold = True; current_stop = entry_price
        if candle['low'] <= current_stop: final_pnl = current_stop - entry_price; exit_date = date; break
        if candle['close'] > entry_price:
            new_stop = max(current_stop, entry_price)
            if cfg['use_aggressive_breakeven'] and not leg1_sold and not aggressive_be_triggered:
                new_stop = max(new_stop, entry_price + cfg['breakeven_buffer_points']); aggressive_be_triggered = True
            if cfg['trail_on_htf_low_after_be']:
                start_of_week = date - pd.to_timedelta(date.weekday(), unit='D')
                new_stop = max(new_stop, df.loc[start_of_week:date]['low'].min())
            elif candle['green_candle']: new_stop = max(new_stop, candle['low'])
            current_stop = new_stop
    if exit_date is None: exit_date, final_pnl = df.index[-1], df.iloc[-1]['close'] - entry_price
    total_pnl = (partial_exit_pnl * 0.5) + (final_pnl * 0.5) if leg1_sold else final_pnl
    return exit_date, total_pnl

def run_backtest(cfg):
    start_time = time.time()
    base_timeframe = cfg['timeframe'].replace('-immediate', '')
    data_folder_htf, data_folder_daily, intraday_folder = os.path.join(cfg['data_folder_base'], base_timeframe), os.path.join(cfg['data_folder_base'], 'daily'), cfg['intraday_data_folder']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); os.makedirs(cfg['log_folder'], exist_ok=True)
    try: symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError: print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return
    print("Loading all Daily, HTF, and 15-min data into memory...")
    daily_data, intraday_data, htf_data = {}, {}, {}
    index_df_daily = None
    for symbol in symbols + ['NIFTY200_INDEX', 'INDIAVIX']:
        try:
            daily_file, htf_file = os.path.join(data_folder_daily, f"{symbol}_daily_with_indicators.csv"), os.path.join(data_folder_htf, f"{symbol}_{base_timeframe}_with_indicators.csv")
            if "NIFTY200_INDEX" in symbol: intraday_filename = "NIFTY_200_15min.csv"
            elif "INDIAVIX" in symbol: intraday_filename = None
            else: intraday_filename = f"{symbol}_15min.csv"
            if os.path.exists(daily_file):
                df_d = pd.read_csv(daily_file, index_col='datetime', parse_dates=True)
                df_d.rename(columns=lambda x: x.lower(), inplace=True); df_d['red_candle'], df_d['green_candle'] = df_d['close'] < df_d['open'], df_d['close'] > df_d['open']; daily_data[symbol] = df_d
                if "NIFTY200_INDEX" in symbol: index_df_daily = df_d
            if os.path.exists(htf_file):
                df_h = pd.read_csv(htf_file, index_col='datetime', parse_dates=True)
                df_h.rename(columns=lambda x: x.lower(), inplace=True); df_h['red_candle'], df_h['green_candle'] = df_h['close'] < df_h['open'], df_h['close'] > df_h['open']; htf_data[symbol] = df_h
            if intraday_filename and os.path.exists(os.path.join(intraday_folder, intraday_filename)):
                intraday_data[symbol] = pd.read_csv(os.path.join(intraday_folder, intraday_filename), index_col='datetime', parse_dates=True)
        except Exception as e: print(f"Warning: Could not load all data for {symbol}. Error: {e}")
    print("Data loading complete.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    all_setups_log = []; target_list = {}
    master_dates = daily_data.get('NIFTY200_INDEX', pd.DataFrame()).loc[cfg['start_date']:cfg['end_date']].index
    
    print("Starting HTF HYBRID backtest simulation (Scout & Sniper)...")
    for date in master_dates:
        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])} | Sniper Targets: {len(target_list.get(date, {}))}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()
        
        # --- SNIPER (Day T): Execute trades for setups identified by the Scout on Day T-1 ---
        if date in target_list:
            equity_at_sod = portfolio['equity']
            sniper_watchlist = target_list.pop(date)
            try:
                today_intraday_candles = intraday_data.get('NIFTY200_INDEX', pd.DataFrame()).loc[date.date().strftime('%Y-%m-%d')]
                if not today_intraday_candles.empty:
                    htf_period_openers = {}
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

                        for symbol, details in list(sniper_watchlist.items()):
                            if symbol not in intraday_data or candle_time not in intraday_data[symbol].index or any(p['symbol'] == symbol for p in portfolio['positions'].values()): continue
                            filters_passed = True
                            candle = intraday_data[symbol].loc[candle_time]
                            if cfg['prevent_entry_below_trigger'] and candle['close'] < details['trigger_price']: filters_passed = False
                            if filters_passed:
                                entry_price = candle['close']
                                daily_df, daily_loc = daily_data[symbol], daily_data[symbol].index.get_loc(date)
                                stop_loss = daily_df.iloc[max(0, daily_loc - cfg['stop_loss_lookback']):daily_loc]['low'].min()
                                if pd.isna(stop_loss): continue
                                risk_per_share = entry_price - stop_loss
                                if risk_per_share <= 0: continue
                                shares = math.floor((equity_at_sod * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)
                                log_entry = {'setup_id': details['setup_id'], 'symbol': symbol, 'setup_date': details['setup_date'], 'trigger_price': details['trigger_price'], 'status': ''}
                                if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                                    log_entry['status'] = 'FILLED'; portfolio['cash'] -= shares * entry_price
                                    portfolio['positions'][f"{symbol}_{candle_time}"] = {'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price, 'stop_loss': stop_loss, 'shares': shares, 'target': entry_price + risk_per_share, 'partial_exit': False, 'initial_shares': shares, 'initial_stop_loss': stop_loss, 'setup_id': details['setup_id']}
                                elif shares > 0: log_entry['status'] = 'MISSED_CAPITAL'
                                all_setups_log.append(log_entry)
                                sniper_watchlist.pop(symbol)
            except KeyError: pass

        # --- SCOUT (End of Day T): Find setups for Day T+1 ---
        next_day = date + pd.Timedelta(days=1)
        if next_day not in master_dates: continue
        market_uptrend = True
        if cfg['market_regime_filter'] and date in index_df_daily.index:
            if index_df_daily.loc[date]['close'] < index_df_daily.loc[date][f"ema_{cfg['regime_ma_period']}"]: market_uptrend = False
        if market_uptrend:
            for symbol in symbols:
                if any(p['symbol'] == symbol for p in portfolio['positions'].values()) or symbol not in htf_data or symbol not in daily_data or date not in daily_data[symbol].index: continue
                df_htf, df_d = htf_data[symbol], daily_data[symbol]
                try:
                    htf_loc = df_htf.index.searchsorted(date, side='right') - 1
                    if htf_loc < 1: continue
                    red_candles_h = get_consecutive_red_candles_htf(df_htf, htf_loc)
                    if not red_candles_h: continue
                    trigger_price = max([c['high'] for c in red_candles_h])
                    daily_candle = df_d.loc[date]
                    if daily_candle['high'] >= trigger_price and daily_candle['open'] < trigger_price:
                        start_of_htf_period = df_htf.index[htf_loc]
                        days_in_period = df_d.loc[start_of_htf_period:date]
                        if not days_in_period.empty and days_in_period.iloc[:-1]['high'].max() >= trigger_price: continue
                        rs_ok = not cfg['rs_filter'] or (date in index_df_daily.index and df_d.loc[date, f"return_{cfg['rs_period']}"] > index_df_daily.loc[date, f"return_{cfg['rs_period']}"])
                        if not rs_ok: continue
                        volume_ok = not cfg['volume_filter'] or (pd.notna(daily_candle[f"volume_{cfg['volume_ma_period']}_sma"]) and daily_candle['volume'] >= (daily_candle[f"volume_{cfg['volume_ma_period']}_sma"] * cfg['volume_multiplier']))
                        if not volume_ok: continue
                        htf_setup_date = df_htf.index[htf_loc]
                        setup_id = f"{symbol}_{htf_setup_date.strftime('%Y-%m-%d')}"
                        if next_day not in target_list: target_list[next_day] = {}
                        target_list[next_day][symbol] = {'setup_id': setup_id, 'setup_date': htf_setup_date, 'trigger_price': trigger_price}
                        all_setups_log.append({'setup_id': setup_id, 'symbol': symbol, 'setup_date': htf_setup_date, 'trigger_date': date, 'trigger_price': trigger_price, 'status': 'IDENTIFIED'})
                except (KeyError, IndexError): continue

        # --- EOD Position and Equity Management ---
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                daily_candle = daily_data[pos['symbol']].loc[date]
                new_stop = pos['stop_loss']
                if daily_candle['close'] > pos['entry_price']:
                    new_stop = max(new_stop, pos['entry_price'])
                    if daily_candle['green_candle']: new_stop = max(new_stop, daily_candle['low'])
                pos['stop_loss'] = new_stop
        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                eod_equity += pos['shares'] * daily_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    # --- Final Reporting ---
    print("\n--- BACKTEST COMPLETE ---")
    for log_entry in all_setups_log:
        if log_entry.get('status') in ['CANCELLED_GAP_UP', 'CANCELLED_FAILED_BREAKOUT', 'IDENTIFIED']: continue
        if 'symbol' not in log_entry or 'setup_date' not in log_entry: continue
        daily_df = daily_data[log_entry['symbol']]
        if log_entry['setup_date'] not in daily_df.index: continue
        loc = daily_df.index.get_loc(log_entry['setup_date'])
        stop_loss = daily_df.iloc[max(0, loc - cfg['stop_loss_lookback']):loc]['low'].min()
        if pd.isna(stop_loss): continue
        exit_date, pnl = simulate_trade_outcome(log_entry['symbol'], log_entry['setup_date'], log_entry['trigger_price'], stop_loss, daily_data, cfg)
        log_entry['hypothetical_exit_date'], log_entry['hypothetical_pnl_per_share'] = exit_date, pnl
    
    final_equity = portfolio['equity']
    net_pnl = final_equity - cfg['initial_capital']
    equity_df = pd.DataFrame(portfolio['daily_values']).set_index('date')
    
    if not equity_df.empty:
        years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = ((final_equity / cfg['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100
    else: cagr, max_drawdown = 0, 0
    
    trades_df = pd.DataFrame(portfolio['trades'])
    total_trades, win_rate, profit_factor, avg_win, avg_loss = 0, 0, 0, 0, 0
    hypothetical_win_rate, hypothetical_profit_factor = 0, 0
    all_setups_df = pd.DataFrame(all_setups_log)
    
    if not trades_df.empty:
        winning_trades, losing_trades = trades_df[trades_df['pnl'] > 0], trades_df[trades_df['pnl'] <= 0]
        total_trades = len(trades_df)
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        gross_profit, gross_loss = winning_trades['pnl'].sum(), abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win, avg_loss = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0, abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        
    if not all_setups_df.empty:
        valid_setups_df = all_setups_df[all_setups_df['status'].isin(['FILLED', 'MISSED_CAPITAL'])]
        filled_trades_for_hypo = trades_df[trades_df['exit_type'] != 'Partial Profit (1:1)'].copy()
        if 'initial_shares' in filled_trades_for_hypo.columns and not filled_trades_for_hypo.empty:
            filled_trades_for_hypo = filled_trades_for_hypo[filled_trades_for_hypo['initial_shares'] > 0]
            filled_trades_for_hypo['pnl_per_share'] = filled_trades_for_hypo['pnl'] / filled_trades_for_hypo['initial_shares']
        else: filled_trades_for_hypo['pnl_per_share'] = 0
        hypo_pnl_list = list(valid_setups_df[valid_setups_df['status'] == 'MISSED_CAPITAL'].get('hypothetical_pnl_per_share', [])) + list(filled_trades_for_hypo['pnl_per_share'])
        if hypo_pnl_list:
            winning_setups, losing_setups = [p for p in hypo_pnl_list if p > 0], [p for p in hypo_pnl_list if p <= 0]
            hypothetical_win_rate = (len(winning_setups) / len(hypo_pnl_list)) * 100 if hypo_pnl_list else 0
            gross_hypo_profit, gross_hypo_loss = sum(winning_setups), abs(sum(losing_setups))
            hypothetical_profit_factor = gross_hypo_profit / gross_hypo_loss if gross_hypo_loss > 0 else float('inf')

    params_str = "INPUT PARAMETERS:\n-----------------\n"
    for key, value in cfg.items(): params_str += f"{key.replace('_', ' ').title()}: {value}\n"
    summary_content = f"""BACKTEST SUMMARY REPORT (HTF HYBRID)\n================================\n{params_str}\nREALISTIC PERFORMANCE (CAPITAL CONSTRAINED):\n--------------------------------------------\nFinal Equity: {final_equity:,.2f}\nNet P&L: {net_pnl:,.2f}\nCAGR: {cagr:.2f}%\nMax Drawdown: {max_drawdown:.1f}%\nTotal Trade Events (incl. partials): {total_trades}\nWin Rate (of events): {win_rate:.1f}%\nProfit Factor: {profit_factor:.2f}\nAverage Winning Event: {avg_win:,.2f}\nAverage Losing Event: {avg_loss:,.2f}\n\nHYPOTHETICAL PERFORMANCE (UNCONSTRAINED):\n-----------------------------------------\nTotal Setups Found: {len(all_setups_df[all_setups_df['status'] != 'IDENTIFIED'])}\nStrategy Win Rate (per setup): {hypothetical_win_rate:.1f}%\nStrategy Profit Factor (per setup): {hypothetical_profit_factor:.2f}\n"""
    
    summary_filename = os.path.join(cfg['log_folder'], f"{timestamp}_summary_report_htf_hybrid.txt")
    trades_filename = os.path.join(cfg['log_folder'], f"{timestamp}_trades_detail_htf_hybrid.csv")
    all_setups_filename = os.path.join(cfg['log_folder'], f"{timestamp}_all_setups_log_htf_hybrid.csv")
    
    with open(summary_filename, 'w') as f: f.write(summary_content)
    if not trades_df.empty: trades_df.to_csv(trades_filename, index=False)
    if not all_setups_df.empty: all_setups_df.to_csv(all_setups_filename, index=False)

    print(summary_content)
    print(f"Backtest completed in {time.time()-start_time:.2f} seconds")
    print(f"Reports saved to '{cfg['log_folder']}'")


if __name__ == "__main__":
    run_backtest(config)
