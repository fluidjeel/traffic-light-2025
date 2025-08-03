# simulator_daily_hybrid.py
#
# Description:
# This is the state-of-the-art, bias-free backtester for the daily strategy.
#
# MODIFICATION (v3.5 - Final Filtered Log Fix):
# 1. FIXED: A bug where the data for filtered trades was not being collected
#    during the intraday loop.
# 2. FIXED: Restored the missing logic to save the _filtered.csv log file in
#    the final reporting section.
#
# MODIFICATION (v3.3 - Logging Fix):
# 1. FIXED: Restored the missing logic to save the _missed_trades.csv file.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time, timedelta
import time
import sys
import numpy as np

# --- CONFIGURATION ---
config = {
    # --- General Backtest Parameters ---
    'initial_capital': 1000000,
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',

    # --- Data & Logging Paths ---
    'data_folder_base': 'data/processed',
    'intraday_data_folder': 'historical_data_15min',
    'log_folder': 'backtest_logs',
    'strategy_name': 'simulator_daily_hybrid',

    # --- Enhanced Logging Options ---
    'log_options': {
        'log_trades': True,      # If True, saves a CSV of all successfully executed trades.
        'log_missed': True,      # If True, saves a CSV of trades missed due to capital or risk limits.
        'log_summary': True,     # If True, saves the main summary.txt report.
        'log_filtered': True     # If True, saves a CSV of setups that failed intraday conviction filters.
    },

    # --- Core Daily Strategy Parameters (EOD Scan) ---
    'timeframe': 'daily',
    'ema_period': 30,
    'rs_period': 30,

    # --- Stop-Loss Configuration ---
    'stop_loss_mode': 'LOOKBACK',           # Options: 'LOOKBACK' or 'PERCENT'.
    'fixed_stop_loss_percent': 0.09,        # Child of 'stop_loss_mode':'PERCENT'.
    'stop_loss_lookback_days': 5,           # Child of 'stop_loss_mode':'LOOKBACK'.

    # --- Trade Management & Profit Taking ---
    'use_partial_profit_leg': False,
    'use_aggressive_breakeven': True,
    'breakeven_buffer_points': 0.05,
    'profit_target_multiplier': 1.2,

    # --- EOD Filters (For Watchlist Generation) ---
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200_INDEX',
    'regime_ma_period': 50,
    'volume_filter': True,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    'rs_filter': True,

    # --- Conviction Engine (Intraday Sniper Filters) ---
    'use_dynamic_position_sizing': True,
    'max_portfolio_risk_percent': 16.0,
    'risk_per_trade_percent': 4.0,
    'max_new_positions_per_day': 5,
    
    'use_slippage': True,
    'adaptive_slippage': True,
    'vix_threshold': 25,
    'base_slippage_percent': 0.05,
    'high_vol_slippage_percent': 0.15,
    
    'use_volume_projection': True,
    'volume_surge_multiplier': 2.0,
    'volume_acceleration_factor': 1.2,
    'volume_projection_thresholds': { dt_time(10, 0): 0.20, dt_time(11, 30): 0.45, dt_time(13, 0): 0.65, dt_time(14, 0): 0.85 },

    'use_enhanced_market_strength': True,
    'market_strength_threshold': -0.15,
    'vix_symbol': 'INDIAVIX',
}


def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 1
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def check_volume_projection(candle_time, cumulative_volume, target_daily_volume, thresholds, volume_surge_credit, cfg):
    current_time = candle_time.time()
    applicable_threshold = 0
    for threshold_time, threshold_pct in sorted(thresholds.items()):
        if current_time >= threshold_time:
            applicable_threshold = threshold_pct
        else:
            break
    if applicable_threshold == 0:
        return True, 0, 0, 0

    required_volume = target_daily_volume * applicable_threshold
    if volume_surge_credit:
        required_volume *= cfg['volume_acceleration_factor']
    return cumulative_volume >= required_volume, cumulative_volume, required_volume, applicable_threshold

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
    strategy_log_folder = os.path.join(cfg['log_folder'], cfg['strategy_name'])
    os.makedirs(strategy_log_folder, exist_ok=True)

    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return

    print("Loading all Daily and 15-min data into memory...")
    daily_data, intraday_data = {}, {}
    index_df_daily = None
    symbols_to_load = symbols + [cfg['regime_index_symbol'], cfg['vix_symbol']]
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
            print(f"Warning: Could not load all data for {symbol}. Error: {e}")
    print("Data loading complete.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    all_setups_log = []
    watchlist = {}
    filtered_log = []
    debug_log = []

    master_dates = index_df_daily.loc[cfg['start_date']:cfg['end_date']].index

    print("Starting Daily Hybrid Simulator with Advanced Conviction Engine...")
    for date in master_dates:
        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])} | Watchlist: {len(watchlist.get(date, {}))}"
        sys.stdout.write(f"\r{progress_str.ljust(120)}"); sys.stdout.flush()

        equity_at_sod = portfolio['equity']
        todays_watchlist = watchlist.get(date, {})
        todays_new_positions = 0

        precomputed_data = {}
        for symbol, details in todays_watchlist.items():
            if symbol in intraday_data:
                try:
                    df_intra = intraday_data[symbol].loc[date.date().strftime('%Y-%m-%d')]
                    precomputed_data[symbol] = {
                        'cum_vol_series': df_intra['volume'].cumsum(),
                        'intraday_candles': df_intra
                    }
                except KeyError:
                    continue

        try:
            today_intraday_candles = intraday_data.get(cfg['regime_index_symbol'], pd.DataFrame()).loc[date.date().strftime('%Y-%m-%d')]
        except KeyError: today_intraday_candles = pd.DataFrame()

        if not today_intraday_candles.empty and todays_watchlist:
            try:
                vix_df = daily_data[cfg['vix_symbol']]
                current_date_pos_indexer = vix_df.index.get_indexer([date], method='ffill')
                if current_date_pos_indexer[0] > 0:
                    prev_day_loc = current_date_pos_indexer[0] - 1
                    vix_prev_close = vix_df.iloc[prev_day_loc]['close']
                else:
                    vix_prev_close = 15
            except (KeyError, IndexError):
                vix_prev_close = 15

            for candle_time in today_intraday_candles.index:
                exit_proceeds, to_remove = 0, []
                for pos_id, pos in list(portfolio['positions'].items()):
                    if pos['symbol'] not in intraday_data or candle_time not in intraday_data[pos['symbol']].index: continue
                    candle = intraday_data[pos['symbol']].loc[candle_time]

                    pos['lowest_price_since_entry'] = min(pos.get('lowest_price_since_entry', pos['entry_price']), candle['low'])
                    
                    if cfg['use_partial_profit_leg'] and not pos.get('partial_exit', False) and candle['high'] >= pos['target']:
                        shares, price, exit_type = pos['shares'] // 2, pos['target'], 'Partial Profit'
                        exit_proceeds += shares * price
                        mae_price = pos['lowest_price_since_entry']
                        mae_percent = ((pos['entry_price'] - mae_price) / pos['entry_price']) * 100
                        trade_log = pos.copy(); trade_log.update({'exit_date': candle_time, 'exit_price': price, 'pnl': (price - pos['entry_price']) * shares, 'exit_type': exit_type, 'mae_price': mae_price, 'mae_percent': mae_percent})
                        portfolio['trades'].append(trade_log)
                        pos['shares'] -= shares; pos['partial_exit'] = True; pos['stop_loss'] = pos['entry_price']

                    if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                        price, exit_type = pos['stop_loss'], 'Stop-Loss'
                        exit_proceeds += pos['shares'] * price
                        mae_price = pos['lowest_price_since_entry']
                        mae_percent = ((pos['entry_price'] - mae_price) / pos['entry_price']) * 100
                        trade_log = pos.copy(); trade_log.update({'exit_date': candle_time, 'exit_price': price, 'pnl': (price - pos['entry_price']) * pos['shares'], 'exit_type': exit_type, 'mae_price': mae_price, 'mae_percent': mae_percent})
                        portfolio['trades'].append(trade_log)
                        to_remove.append(pos_id)
                
                for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
                portfolio['cash'] += exit_proceeds

                for symbol, details in list(todays_watchlist.items()):
                    if symbol not in precomputed_data or candle_time not in precomputed_data[symbol]['intraday_candles'].index or any(p['symbol'] == symbol for p in portfolio['positions'].values()): continue

                    candle = precomputed_data[symbol]['intraday_candles'].loc[candle_time]
                    if candle['close'] >= details['trigger_price']:
                        
                        log_template = {'symbol': symbol, 'timestamp': candle_time, 'trigger_price': details['trigger_price']}

                        # --- BUG FIX: Added logic to capture filtered trades ---
                        if cfg['use_volume_projection']:
                            cum_vol_series = precomputed_data[symbol]['cum_vol_series']
                            cumulative_volume = cum_vol_series[cum_vol_series.index <= candle_time].iloc[-1]
                            target_daily_volume = details['target_volume']
                            volume_surge = False
                            current_idx = precomputed_data[symbol]['intraday_candles'].index.get_loc(candle_time)
                            if current_idx >= 3:
                                volume_window = precomputed_data[symbol]['intraday_candles']['volume'].iloc[current_idx-3:current_idx]
                                if len(volume_window) > 0 and volume_window.median() > 0 and candle['volume'] > volume_window.median() * cfg['volume_surge_multiplier']:
                                    volume_surge = True
                            vol_ok, cum_vol, req_vol, _ = check_volume_projection(candle_time, cumulative_volume, target_daily_volume, cfg['volume_projection_thresholds'], volume_surge, cfg)
                            if not vol_ok:
                                filtered_log.append({**log_template, 'filter_type': 'Volume Projection', 'actual': f"{cum_vol:,.0f}", 'expected': f"{req_vol:,.0f}"})
                                del todays_watchlist[symbol] # Remove from watchlist to avoid re-checking
                                continue

                        if cfg['use_enhanced_market_strength']:
                            threshold = cfg['market_strength_threshold'] * (1.5 if vix_prev_close > cfg['vix_threshold'] else 1.0)
                            strength_score = (today_intraday_candles.loc[candle_time]['close'] / today_intraday_candles.iloc[0]['open'] - 1) * 100
                            if strength_score < threshold:
                                filtered_log.append({**log_template, 'filter_type': 'Market Strength', 'actual': f"{strength_score:.2f}%", 'expected': f">{threshold:.2f}%"})
                                del todays_watchlist[symbol]
                                continue
                        # --- END BUG FIX ---

                        entry_price_base = candle['close']
                        slippage = 0
                        if cfg['use_slippage']:
                            slippage_pct = cfg['high_vol_slippage_percent'] if cfg['adaptive_slippage'] and vix_prev_close > cfg['vix_threshold'] else cfg['base_slippage_percent']
                            slippage = entry_price_base * (slippage_pct / 100)
                        entry_price = entry_price_base + slippage

                        if cfg.get('stop_loss_mode') == 'PERCENT':
                            stop_loss = entry_price * (1 - cfg['fixed_stop_loss_percent'])
                        else:
                            daily_df, daily_loc = daily_data[symbol], daily_data[symbol].index.get_loc(date)
                            stop_loss = daily_df.iloc[max(0, daily_loc - cfg['stop_loss_lookback_days']):daily_loc]['low'].min()
                        
                        if pd.isna(stop_loss): continue
                        risk_per_share = entry_price - stop_loss
                        if risk_per_share <= 0: continue

                        if cfg['use_dynamic_position_sizing']:
                            active_risk = sum([(p['entry_price'] - p['stop_loss']) * p['shares'] for p in portfolio['positions'].values()])
                            max_total_risk = equity_at_sod * (cfg['max_portfolio_risk_percent'] / 100)
                            available_risk_capital = max(0, max_total_risk - active_risk)
                            risk_per_trade = equity_at_sod * (cfg['risk_per_trade_percent'] / 100)
                            capital_for_this_trade = min(available_risk_capital, risk_per_trade)
                            shares = math.floor(capital_for_this_trade / risk_per_share) if capital_for_this_trade > 0 else 0
                        else:
                            shares = math.floor((equity_at_sod * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)

                        setup_log_entry = next((item for item in all_setups_log if item['setup_id'] == details['setup_id']), None)
                        if setup_log_entry:
                            if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                                setup_log_entry['status'] = 'FILLED'
                                portfolio['cash'] -= shares * entry_price
                                todays_new_positions += 1
                                profit_multiplier = cfg['profit_target_multiplier']
                                if cfg['adaptive_slippage'] and vix_prev_close > cfg['vix_threshold']:
                                    profit_multiplier = 1.0
                                portfolio['positions'][f"{symbol}_{candle_time}"] = {
                                    'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price,
                                    'stop_loss': stop_loss, 'shares': shares,
                                    'target': entry_price + (risk_per_share * profit_multiplier),
                                    'partial_exit': False, 'initial_shares': shares, 'setup_id': details['setup_id'],
                                    'lowest_price_since_entry': entry_price
                                }
                            elif shares > 0:
                                setup_log_entry['status'] = 'MISSED_CAPITAL'
                        
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
                    setup_candle = df.iloc[loc]
                    
                    if not (setup_candle['green_candle'] and setup_candle['close'] > setup_candle[f"ema_{cfg['ema_period']}"]):
                        continue
                    
                    red_candles = get_consecutive_red_candles(df, loc)
                    if not red_candles: continue

                    rs_ok = not cfg['rs_filter'] or (date in index_df_daily.index and pd.notna(df.loc[date, f"return_{cfg['rs_period']}"]) and pd.notna(index_df_daily.loc[date, f"return_{cfg['rs_period']}"]) and df.loc[date, f"return_{cfg['rs_period']}"] > index_df_daily.loc[date, f"return_{cfg['rs_period']}"])
                    if not rs_ok: continue

                    volume_ok = not cfg['volume_filter'] or (pd.notna(setup_candle[f"volume_{cfg['volume_ma_period']}_sma"]) and setup_candle['volume'] >= (setup_candle[f"volume_{cfg['volume_ma_period']}_sma"] * cfg['volume_multiplier']))
                    if not volume_ok: continue

                    trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])
                    setup_id = f"{symbol}_{date.strftime('%Y-%m-%d')}"
                    all_setups_log.append({'setup_id': setup_id, 'symbol': symbol, 'setup_date': date, 'trigger_price': trigger_price, 'status': 'IDENTIFIED'})
                    target_volume = setup_candle[f"volume_{cfg['volume_ma_period']}_sma"] * cfg['volume_multiplier']
                    next_day = date + timedelta(days=1)
                    if next_day in master_dates:
                        if next_day not in watchlist: watchlist[next_day] = {}
                        watchlist[next_day][symbol] = {'trigger_price': trigger_price, 'setup_id': setup_id, 'target_volume': target_volume}
                except (KeyError, IndexError): continue

        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                daily_candle = daily_data[pos['symbol']].loc[date]
                new_stop = pos['stop_loss']
                if daily_candle['close'] > pos['entry_price']:
                    new_stop = max(new_stop, pos['entry_price'])
                    if cfg['use_aggressive_breakeven'] and not pos.get('partial_exit', False):
                        new_stop = max(new_stop, pos['entry_price'] + cfg['breakeven_buffer_points'])
                    if daily_candle['green_candle']: new_stop = max(new_stop, daily_candle['low'])
                pos['stop_loss'] = new_stop
        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                eod_equity += pos['shares'] * daily_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    print("\n\n--- BACKTEST COMPLETE ---")
    
    trades_df = pd.DataFrame(portfolio['trades'])
    final_equity = portfolio['equity']
    
    print("\n--- Generating Enhanced Log Files ---")
    log_opts = cfg.get('log_options', {})

    if log_opts.get('log_summary', True):
        params_str = "INPUT PARAMETERS:\n-----------------\n"
        cfg_to_log = {k: v for k, v in cfg.items() if k != 'use_ema_filter'}
        for key, value in cfg_to_log.items():
            if isinstance(value, dict):
                params_str += f"{key.replace('_', ' ').title()}:\n"
                for k, v in value.items(): params_str += f"  - {k}: {v}\n"
            else:
                params_str += f"{key.replace('_', ' ').title()}: {value}\n"
        
        net_pnl = final_equity - cfg['initial_capital']
        equity_df = pd.DataFrame(portfolio['daily_values']).set_index('date')
        cagr, max_drawdown = 0, 0
        if not equity_df.empty:
            years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
            cagr = ((final_equity / cfg['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
            peak = equity_df['equity'].cummax(); drawdown = (equity_df['equity'] - peak) / peak; max_drawdown = abs(drawdown.min()) * 100
        
        total_trades, win_rate, profit_factor = 0, 0, 0
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            total_trades = len(trades_df)
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            gross_profit = winning_trades['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        summary_content = f"""BACKTEST SUMMARY REPORT ({cfg['strategy_name'].upper()})\n===================================================================\n{params_str}
ACTUAL TRADES PERFORMANCE (CAPITAL CONSTRAINED):
------------------------------------------------
Final Equity: {final_equity:,.2f}
Net P&L: {net_pnl:,.2f}
CAGR: {cagr:.2f}%
Max Drawdown: {max_drawdown:.1f}%
Total Trade Events: {total_trades}
Win Rate: {win_rate:.1f}%
Profit Factor: {profit_factor:.2f}
"""
        summary_filename = os.path.join(strategy_log_folder, f"{timestamp}_summary.txt")
        with open(summary_filename, 'w') as f: f.write(summary_content)
        print(f"Summary report saved to '{summary_filename}'")

    if log_opts.get('log_trades', True) and not trades_df.empty:
        trades_filename = os.path.join(strategy_log_folder, f"{timestamp}_trade_details.csv")
        trades_df.to_csv(trades_filename, index=False)
        print(f"Trade details saved to '{trades_filename}'")

    if log_opts.get('log_missed', True) and all_setups_log:
        all_setups_df = pd.DataFrame(all_setups_log)
        missed_trades_df = all_setups_df[all_setups_df['status'] == 'MISSED_CAPITAL'].copy()
        if not missed_trades_df.empty:
            missed_filename = os.path.join(strategy_log_folder, f"{timestamp}_missed_trades.csv")
            missed_trades_df.to_csv(missed_filename, index=False)
            print(f"Missed trades log saved to '{missed_filename}'")
    
    if log_opts.get('log_filtered', True) and filtered_log:
        filtered_df = pd.DataFrame(filtered_log)
        filtered_filename = os.path.join(strategy_log_folder, f"{timestamp}_filtered.csv")
        filtered_df.to_csv(filtered_filename, index=False)
        print(f"Filtered setups log saved to '{filtered_filename}'")
    
if __name__ == "__main__":
    run_backtest(config)
