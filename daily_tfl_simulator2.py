# daily_tfl_simulator.py
#
# Description:
# A complete and corrected version of the bias-free backtester for the daily strategy.
# This version includes VIX-scaled profit targets, aggressive breakeven logic,
# and full, detailed logging for MAE/MFE analysis.
#
# MODIFICATION (v2.7 - Logic & Bug Fixes):
# 1. FIXED: The entry trigger condition now correctly uses the 15-minute candle's 'high'
#    instead of 'close', ensuring breakouts are not missed.
# 2. FIXED: The entry price is now based on the 'trigger_price' itself, simulating a
#    more realistic fill for a breakout order.
# 3. FIXED: The final summary report generation logic to be complete and accurate.
# 4. REMOVED: Unused 'rs_period' from the config to avoid confusion.

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

    # --- DATA PIPELINE CONFIGURATION ---
    'data_pipeline_config': {
        'use_universal_pipeline': True,
        'universal_processed_folder': os.path.join('data', 'universal_processed'),
        'universal_intraday_folder': os.path.join('data', 'universal_historical_data'),
    },

    # --- Logging & Strategy Info ---
    'log_folder': 'backtest_logs',
    'strategy_name': 'daily_tfl_simulator',
    'log_options': { 'log_trades': True, 'log_missed': True, 'log_summary': True, 'log_filtered': True },

    # --- Core Daily Strategy Parameters (EOD Scan) ---
    'timeframe': 'daily',
    'ema_period': 30,
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200_INDEX',
    'regime_ma_period': 50,
    'volume_filter': True,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,

    # --- Stop-Loss & Trade Management ---
    'stop_loss_mode': 'LOOKBACK',
    'stop_loss_lookback_days': 3,
    'use_aggressive_breakeven': True,
    'breakeven_buffer_percent': 0.0005,
    'use_partial_profit_leg': True,

    # --- VIX-Scaled Profit Target (Data-Driven from Analyzer) ---
    'vix_scaled_profit_target': {
        'use_vix_scaling': True,
        'scale': [
            {'vix_max': 15, 'multiplier': 2.5},  # Calm Market
            {'vix_max': 22, 'multiplier': 3.0},  # Moderate Volatility
            {'vix_max': 999, 'multiplier': 3.5} # High Volatility
        ]
    },

    # --- Risk & Position Sizing ---
    'use_dynamic_position_sizing': True,
    'max_portfolio_risk_percent': 6.0,
    'risk_per_trade_percent': 2.0,
    'max_new_positions_per_day': 5,
    
    # --- Slippage & VIX Settings ---
    'use_slippage': True,
    'adaptive_slippage': True,
    'vix_threshold': 25,
    'base_slippage_percent': 0.05,
    'high_vol_slippage_percent': 0.15,
    'vix_symbol': 'INDIAVIX',
}

# --- HELPER FUNCTIONS ---
def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 1
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def create_enhanced_trade_log(pos, exit_time, exit_price, exit_type, shares_exited):
    base_log = pos.copy()
    base_log.update({
        'exit_date': exit_time, 'exit_price': exit_price,
        'pnl': (exit_price - pos['entry_price']) * shares_exited, 'exit_type': exit_type,
    })
    
    mae_price = pos['lowest_price_since_entry']
    mfe_price = pos['highest_price_since_entry']
    
    if pos['entry_price'] > 0:
        mae_percent = ((pos['entry_price'] - mae_price) / pos['entry_price']) * 100
        mfe_percent = ((mfe_price - pos['entry_price']) / pos['entry_price']) * 100
        captured_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
    else:
        mae_percent, mfe_percent, captured_pct = 0, 0, 0
        
    base_log.update({
        'mae_price': mae_price, 'mae_percent': mae_percent,
        'mfe_price': mfe_price, 'mfe_percent': mfe_percent,
        'captured_pct': captured_pct,
    })
    return base_log

# --- MAIN SIMULATOR ---
def run_backtest(cfg):
    start_time = time.time()
    
    cfg['data_folder_base'] = cfg['data_pipeline_config']['universal_processed_folder']
    cfg['intraday_data_folder'] = cfg['data_pipeline_config']['universal_intraday_folder']
    daily_folder = os.path.join(cfg['data_folder_base'], 'daily')
    intraday_folder = cfg['intraday_data_folder']
    
    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return

    print("Loading all data into memory...")
    daily_data, intraday_data = {}, {}
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
            
            intraday_file = os.path.join(intraday_folder, f"{symbol}_15min.csv")
            if os.path.exists(intraday_file):
                intraday_data[symbol] = pd.read_csv(intraday_file, index_col='datetime', parse_dates=True)
        except Exception as e:
            print(f"Warning: Could not load data for {symbol}. Error: {e}")
    print("Data loading complete.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    watchlist = {}
    all_setups_log = []
    filtered_log = []
    
    master_dates = daily_data[cfg['regime_index_symbol']].loc[cfg['start_date']:cfg['end_date']].index

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_log_folder = os.path.join(cfg['log_folder'], cfg['strategy_name'])
    os.makedirs(strategy_log_folder, exist_ok=True)

    print("Starting Daily Simulator...")
    for date in master_dates:
        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()

        todays_watchlist = watchlist.get(date, {})
        if todays_watchlist:
            try:
                vix_df = daily_data[cfg['vix_symbol']]
                vix_close_t1 = vix_df.loc[:date].iloc[-2]['close']
                today_intraday_candles = intraday_data.get(cfg['regime_index_symbol'], pd.DataFrame()).loc[date.date().strftime('%Y-%m-%d')]
            except (KeyError, IndexError): continue

            if not today_intraday_candles.empty:
                for candle_time, _ in today_intraday_candles.iterrows():
                    exit_proceeds, to_remove = 0, []
                    for pos_id, pos in list(portfolio['positions'].items()):
                        if pos['symbol'] not in intraday_data or candle_time not in intraday_data[pos['symbol']].index: continue
                        candle = intraday_data[pos['symbol']].loc[candle_time]

                        pos['lowest_price_since_entry'] = min(pos.get('lowest_price_since_entry', pos['entry_price']), candle['low'])
                        pos['highest_price_since_entry'] = max(pos.get('highest_price_since_entry', pos['entry_price']), candle['high'])

                        if 'target' in pos and not pos.get('partial_exit', False) and candle['high'] >= pos['target']:
                            exit_price = pos['target']
                            if candle['open'] >= pos['target']: exit_price = candle['open']
                            if cfg['use_partial_profit_leg']:
                                shares_to_sell = pos['shares'] // 2
                                trade_log = create_enhanced_trade_log(pos, candle_time, exit_price, 'Partial Profit', shares_to_sell)
                                exit_proceeds += shares_to_sell * exit_price
                                pos['shares'] -= shares_to_sell
                                pos['partial_exit'] = True
                                pos['stop_loss'] = pos['entry_price']
                            else:
                                trade_log = create_enhanced_trade_log(pos, candle_time, exit_price, 'Profit Target', pos['shares'])
                                exit_proceeds += pos['shares'] * exit_price
                                to_remove.append(pos_id)
                            portfolio['trades'].append(trade_log)
                            if not cfg['use_partial_profit_leg']: continue

                        if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                            exit_price = pos['stop_loss']
                            if candle['open'] <= pos['stop_loss']: exit_price = candle['open']
                            trade_log = create_enhanced_trade_log(pos, candle_time, exit_price, 'Stop-Loss', pos['shares'])
                            portfolio['trades'].append(trade_log)
                            exit_proceeds += pos['shares'] * exit_price
                            to_remove.append(pos_id)
                    
                    for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
                    portfolio['cash'] += exit_proceeds

                    for symbol, details in list(todays_watchlist.items()):
                        if symbol not in intraday_data or candle_time not in intraday_data[symbol].index or any(p['symbol'] == symbol for p in portfolio['positions'].values()): continue
                        
                        candle = intraday_data[symbol].loc[candle_time]
                        
                        ### BUG FIX: Use 'high' for breakout trigger, not 'close' ###
                        if candle['high'] >= details['trigger_price']:
                            ### BUG FIX: Use 'trigger_price' for entry base, not 'close' ###
                            entry_price_base = details['trigger_price']
                            slippage_pct = cfg['high_vol_slippage_percent'] if cfg['adaptive_slippage'] and vix_close_t1 > cfg['vix_threshold'] else cfg['base_slippage_percent']
                            entry_price = entry_price_base * (1 + (slippage_pct / 100))

                            daily_df = daily_data[symbol]
                            loc = daily_df.index.get_loc(date)
                            stop_loss = daily_df.iloc[max(0, loc - cfg['stop_loss_lookback_days']):loc]['low'].min()
                            
                            if pd.isna(stop_loss): continue
                            risk_per_share = entry_price - stop_loss
                            if risk_per_share <= 0: continue
                            
                            pt_config = cfg.get('vix_scaled_profit_target', {})
                            profit_target_multiplier = 2.0
                            if pt_config.get('use_vix_scaling', False):
                                scale = pt_config.get('scale', [])
                                profit_target_multiplier = scale[-1]['multiplier']
                                for level in scale:
                                    if vix_close_t1 <= level['vix_max']:
                                        profit_target_multiplier = level['multiplier']; break

                            current_cash = portfolio['cash']
                            risk_per_trade = current_cash * (cfg['risk_per_trade_percent'] / 100)
                            shares = math.floor(risk_per_trade / risk_per_share)

                            setup_log_entry = next((item for item in all_setups_log if item['setup_id'] == details['setup_id']), None)
                            if setup_log_entry:
                                if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                                    setup_log_entry['status'] = 'FILLED'
                                    portfolio['cash'] -= shares * entry_price
                                    portfolio['positions'][f"{symbol}_{candle_time}"] = {
                                        'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price,
                                        'stop_loss': stop_loss, 'shares': shares,
                                        'target': entry_price + (risk_per_share * profit_target_multiplier),
                                        'partial_exit': False, 'setup_id': details['setup_id'],
                                        'lowest_price_since_entry': entry_price, 'highest_price_since_entry': entry_price,
                                        'vix_on_entry': vix_close_t1
                                    }
                                elif shares > 0:
                                    setup_log_entry['status'] = 'MISSED_CAPITAL'
                                else:
                                    setup_log_entry['status'] = 'FILTERED_RISK'
                            
                            del todays_watchlist[symbol]

        market_uptrend = True
        if cfg['market_regime_filter'] and date in daily_data[cfg['regime_index_symbol']].index:
            if daily_data[cfg['regime_index_symbol']].loc[date]['close'] < daily_data[cfg['regime_index_symbol']].loc[date][f"ema_{cfg['regime_ma_period']}"]:
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
                    
                    trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])
                    setup_id = f"{symbol}_{date.strftime('%Y-%m-%d')}"
                    all_setups_log.append({'setup_id': setup_id, 'symbol': symbol, 'setup_date': date, 'trigger_price': trigger_price, 'status': 'IDENTIFIED'})
                    
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
                    if cfg['use_aggressive_breakeven'] and not pos.get('partial_exit', False):
                        buffer = pos['entry_price'] * cfg['breakeven_buffer_percent']
                        new_stop = max(new_stop, pos['entry_price'] + buffer)
                    if daily_candle['green_candle']:
                        new_stop = max(new_stop, daily_candle['low'])
                pos['stop_loss'] = new_stop
        
        eod_equity = portfolio['cash'] + sum([p['shares'] * daily_data[p['symbol']].loc[date]['close'] for p in portfolio['positions'].values() if date in daily_data.get(p['symbol'], pd.DataFrame()).index])
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    print("\n\n--- BACKTEST COMPLETE ---")
    
    trades_df = pd.DataFrame(portfolio['trades'])
    final_equity = portfolio['equity']
    
    print("\n--- Generating Enhanced Log Files ---")
    log_opts = cfg.get('log_options', {})

    if log_opts.get('log_summary', True):
        params_str = "INPUT PARAMETERS:\n-----------------\n"
        for key, value in cfg.items():
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
            peak = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - peak) / peak
            max_drawdown = abs(drawdown.min()) * 100
        
        total_trades, win_rate, profit_factor = 0, 0, 0
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            total_trades = len(trades_df)
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            gross_profit = winning_trades['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        summary_content = f"""BACKTEST SUMMARY REPORT ({cfg['strategy_name'].upper()})
===================================================================
{params_str}
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
        missed_trades_df = all_setups_df[all_setups_df['status'].isin(['MISSED_CAPITAL', 'FILTERED_RISK'])].copy()
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
