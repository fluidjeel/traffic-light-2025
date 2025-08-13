# candle_breakout_simulator.py
#
# Description:
# A backtester for a daily candle breakout strategy. This is the main version
# for ongoing development and testing, now with a full suite of filters enabled.
# This version fixes a bug in the column naming for the filtered setups log.
#
# Strategy Logic:
# 1. On day (T), it applies a series of configurable filters (Trend, EMA Distance, Volume, RSI).
# 2. If a setup is rejected, its details are logged for later analysis.
# 3. If all filters pass, it places a "straddle" order for day (T+1).

import pandas as pd
import os
import math
from datetime import datetime
import time
import sys
import numpy as np

# --- CONFIGURATION ---
config = {
    # --- General Backtest Parameters ---
    'initial_capital': 1000000,
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',

    # --- Strategy-Specific Settings ---
    'strategy_name': 'candle_breakout_strategy',
    'target_stock': None, # Set to a symbol like 'RELIANCE' for single stock, or None for universe
    'nifty_list_csv': 'nifty500.csv',
    'timeframe': 'daily',
    'setup_candle_type': 'both',
    'risk_reward_ratio': 1.5, # Only used if all trailing stops are False
    'risk_per_trade_percent': 2.0,
    'allow_shorting': False,

    # --- Entry Filters ---
    'entry_filters': {
        'use_ema_filter': True,
        'ema_period': 10,
        
        'use_ema_distance_filter': True,
        'min_long_ema_dist_pct': 0.0,
        'max_long_ema_dist_pct': 20.0,
        'min_short_ema_dist_pct': -5.0,
        'max_short_ema_dist_pct': -1.0,

        'use_volume_filter': True,
        'min_volume_ratio': 1.25,

        'use_rsi_filter': True,
        'min_rsi': 30.0,
        'max_rsi': 70.0,
        
        'preceding_candle_long': 'any',
        'preceding_candle_short': 'any',
        'filter_green_candle_wick': False,
        'max_green_wick_percent': 10.0,
        'filter_red_candle_wick': False,
        'max_red_wick_percent': 10.0,
        'min_risk_percent': 0.1,
    },

    # --- Trade Management & Trailing Stops ---
    'trade_management': {
        'use_atr': True,
        'use_breakeven': True,
        'use_aggressive': False,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'breakeven_buffer_percent': 0.1,
    },

    # --- Data & Logging ---
    'data_folder': os.path.join('data', 'universal_processed'),
    'log_folder': 'backtest_logs',
    'log_options': { 'log_trades': True, 'log_summary': True, 'log_pnl_by_stock': True, 'log_long_short_trades': True, 'log_filtered_setups': True },
    'vix_symbol': 'INDIAVIX',
}


# --- HELPER FUNCTIONS ---
def create_enhanced_trade_log(pos, exit_time, exit_price, exit_type):
    base_log = pos.copy()
    
    if pos['direction'] == 'long':
        pnl = (exit_price - pos['entry_price']) * pos['shares']
    else: # Short
        pnl = (pos['entry_price'] - exit_price) * pos['shares']

    initial_risk_per_share = abs(pos['entry_price'] - pos['initial_stop_loss'])
    
    if pos['direction'] == 'long':
        mae_price = pos.get('lowest_price_since_entry', pos['entry_price'])
        mfe_price = pos.get('highest_price_since_entry', pos['entry_price'])
        mae_R = (pos['entry_price'] - mae_price) / initial_risk_per_share if initial_risk_per_share > 0 else 0
        mfe_R = (mfe_price - pos['entry_price']) / initial_risk_per_share if initial_risk_per_share > 0 else 0
    else:
        mae_price = pos.get('highest_price_since_entry', pos['entry_price'])
        mfe_price = pos.get('lowest_price_since_entry', pos['entry_price'])
        mae_R = (mae_price - pos['entry_price']) / initial_risk_per_share if initial_risk_per_share > 0 else 0
        mfe_R = (pos['entry_price'] - mfe_price) / initial_risk_per_share if initial_risk_per_share > 0 else 0

    base_log.update({
        'exit_date': exit_time, 'exit_price': exit_price, 'pnl': pnl, 'exit_type': exit_type,
        'mae_R': mae_R, 'mfe_R': mfe_R
    })
    base_log.pop('lowest_price_since_entry', None)
    base_log.pop('highest_price_since_entry', None)
    base_log.pop('breakeven_triggered', None)
    return base_log

# --- REFACTORED: Backtest function for a single symbol ---
def run_backtest_for_symbol(symbol, cfg, all_data, vix_data):
    df = all_data.get(symbol)
    if df is None or df.empty:
        return (pd.DataFrame(), {}, [])

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    watchlist = {}
    filtered_setups = []
    master_dates = df.index
    
    tm_cfg = cfg['trade_management']
    is_trailing_active = tm_cfg.get('use_atr') or tm_cfg.get('use_breakeven') or tm_cfg.get('use_aggressive')

    for i, date in enumerate(master_dates):
        # --- Check Watchlist for New Entries ---
        if symbol in watchlist and date >= watchlist[symbol]['trigger_date']:
            orders = watchlist[symbol]
            if date > orders['trigger_date']:
                watchlist.pop(symbol, None)

            if symbol not in portfolio['positions'] and symbol in watchlist:
                today_candle = df.loc[date]
                long_order = orders.get('long')
                short_order = orders.get('short')
                
                high_triggered = long_order and today_candle['high'] >= long_order['trigger_price']
                low_triggered = short_order and today_candle['low'] <= short_order['trigger_price']
                
                entry_order = None
                if high_triggered and low_triggered:
                    if abs(today_candle['open'] - long_order['trigger_price']) < abs(today_candle['open'] - short_order['trigger_price']):
                        entry_order = long_order
                    else:
                        entry_order = short_order
                elif high_triggered:
                    entry_order = long_order
                elif low_triggered:
                    entry_order = short_order

                if entry_order:
                    entry_price, stop_loss_price = entry_order['trigger_price'], entry_order['stop_loss_price']
                    risk_per_share = abs(entry_price - stop_loss_price)
                    
                    if risk_per_share > (entry_price * (cfg['entry_filters']['min_risk_percent'] / 100)):
                        capital_at_risk = portfolio['equity'] * (cfg['risk_per_trade_percent'] / 100)
                        shares = math.floor(capital_at_risk / risk_per_share) if risk_per_share > 0 else 0
                        
                        if shares > 0:
                            if entry_order['direction'] == 'long' and (shares * entry_price) <= portfolio['cash']:
                                portfolio['cash'] -= shares * entry_price
                            elif entry_order['direction'] == 'short' and (shares * entry_price) <= portfolio['cash']:
                                portfolio['cash'] -= shares * entry_price # Margin
                            else:
                                shares = 0

                            if shares > 0:
                                target_price = np.inf if is_trailing_active else entry_price + (risk_per_share * cfg['risk_reward_ratio'])
                                if entry_order['direction'] == 'short':
                                    target_price = -np.inf if is_trailing_active else entry_price - (risk_per_share * cfg['risk_reward_ratio'])
                                
                                setup_candle = df.loc[entry_order['setup_candle_date']]
                                ema_val = setup_candle.get(f"ema_{cfg['entry_filters']['ema_period']}", np.nan)
                                
                                vix_on_entry_value = np.nan
                                if not vix_data.empty and entry_order['setup_candle_date'] in vix_data.index:
                                    vix_on_entry_value = vix_data.loc[entry_order['setup_candle_date']]['close']

                                portfolio['positions'][symbol] = {
                                    'symbol': symbol, 'direction': entry_order['direction'], 'entry_date': date, 'entry_price': entry_price, 'shares': shares,
                                    'stop_loss': stop_loss_price, 'initial_stop_loss': stop_loss_price, 'target': target_price,
                                    'setup_candle_type': entry_order['setup_candle_type'], 'setup_candle_date': entry_order['setup_candle_date'],
                                    'lowest_price_since_entry': entry_price, 'highest_price_since_entry': entry_price, 'breakeven_triggered': False,
                                    'vix_on_entry': vix_on_entry_value,
                                    'rsi_on_entry': setup_candle.get('rsi_14', np.nan),
                                    'volume_ratio_on_entry': setup_candle.get('volume_ratio', np.nan),
                                    'close_to_ema_ratio': (setup_candle['close'] / ema_val - 1) * 100 if pd.notna(ema_val) and ema_val > 0 else np.nan
                                }
                    watchlist.pop(symbol, None)

        # --- Manage Existing Position ---
        if symbol in portfolio['positions']:
            pos = portfolio['positions'][symbol]
            today_candle = df.loc[date]
            pos['lowest_price_since_entry'] = min(pos.get('lowest_price_since_entry', pos['entry_price']), today_candle['low'])
            pos['highest_price_since_entry'] = max(pos.get('highest_price_since_entry', pos['entry_price']), today_candle['high'])

            exit_trade = None
            if pos['direction'] == 'long':
                if today_candle['low'] <= pos['stop_loss']: exit_trade = create_enhanced_trade_log(pos, date, pos['stop_loss'], 'Stop-Loss')
                elif today_candle['high'] >= pos['target']: exit_trade = create_enhanced_trade_log(pos, date, pos['target'], 'Profit Target')
            else:
                if today_candle['high'] >= pos['stop_loss']: exit_trade = create_enhanced_trade_log(pos, date, pos['stop_loss'], 'Stop-Loss')
                elif today_candle['low'] <= pos['target']: exit_trade = create_enhanced_trade_log(pos, date, pos['target'], 'Profit Target')
            
            if exit_trade:
                portfolio['trades'].append(exit_trade)
                if pos['direction'] == 'long': portfolio['cash'] += exit_trade['shares'] * exit_trade['exit_price']
                else: portfolio['cash'] += exit_trade['shares'] * (2 * pos['entry_price'] - exit_trade['exit_price'])
                portfolio['positions'].pop(symbol, None)
            
            else:
                new_stop = pos['stop_loss']
                if date == pos['entry_date'] and tm_cfg.get('use_breakeven'):
                    if pos['direction'] == 'long' and today_candle['close'] > pos['entry_price']:
                        buffer = pos['entry_price'] * (tm_cfg['breakeven_buffer_percent'] / 100)
                        new_stop = max(new_stop, pos['entry_price'] + buffer)
                        pos['breakeven_triggered'] = True
                    elif pos['direction'] == 'short' and today_candle['close'] < pos['entry_price']:
                        buffer = pos['entry_price'] * (tm_cfg['breakeven_buffer_percent'] / 100)
                        new_stop = min(new_stop, pos['entry_price'] - buffer)
                        pos['breakeven_triggered'] = True
                
                elif date > pos['entry_date']:
                    if pos['direction'] == 'long':
                        if tm_cfg.get('use_atr'):
                            atr_col = f"atr_{tm_cfg['atr_period']}"
                            if atr_col in today_candle: new_stop = max(new_stop, pos['highest_price_since_entry'] - (today_candle[atr_col] * tm_cfg['atr_multiplier']))
                        if tm_cfg.get('use_breakeven') and not pos['breakeven_triggered'] and today_candle['close'] > pos['entry_price']:
                            new_stop = max(new_stop, pos['entry_price'] * (1 + tm_cfg['breakeven_buffer_percent'] / 100)); pos['breakeven_triggered'] = True
                        if tm_cfg.get('use_aggressive') and today_candle['green_candle']: new_stop = max(new_stop, today_candle['low'])
                    else: # Short
                        if tm_cfg.get('use_atr'):
                            atr_col = f"atr_{tm_cfg['atr_period']}"
                            if atr_col in today_candle: new_stop = min(new_stop, pos['lowest_price_since_entry'] + (today_candle[atr_col] * tm_cfg['atr_multiplier']))
                        if tm_cfg.get('use_breakeven') and not pos['breakeven_triggered'] and today_candle['close'] < pos['entry_price']:
                            new_stop = min(new_stop, pos['entry_price'] * (1 - tm_cfg['breakeven_buffer_percent'] / 100)); pos['breakeven_triggered'] = True
                        if tm_cfg.get('use_aggressive') and today_candle['red_candle']: new_stop = min(new_stop, today_candle['high'])
                pos['stop_loss'] = new_stop
        
        # --- EOD Equity Update ---
        eod_equity = portfolio['cash']
        if symbol in portfolio['positions']:
            pos = portfolio['positions'][symbol]
            if pos['direction'] == 'long':
                eod_equity += pos['shares'] * df.loc[date]['close']
            else:
                unrealized_pnl = (pos['entry_price'] - df.loc[date]['close']) * pos['shares']
                eod_equity += (pos['entry_price'] * pos['shares']) + unrealized_pnl
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

        # --- Scan for New Setups ---
        next_day_index = i + 1
        if next_day_index < len(master_dates):
            next_day = master_dates[next_day_index]
            if symbol not in portfolio['positions'] and symbol not in watchlist:
                if i == 0: continue
                
                setup_candle = df.loc[date]
                preceding_candle = df.loc[master_dates[i-1]]
                
                ef_cfg = cfg['entry_filters']
                
                # --- MODIFIED: Filter logic now logs rejected setups with correct column names ---
                def log_filtered(reason):
                    ema_val = setup_candle.get(f"ema_{ef_cfg['ema_period']}", np.nan)
                    filtered_setups.append({
                        'symbol': symbol,
                        'setup_date': date,
                        'filter_reason': reason,
                        'vix_on_entry': vix_data.loc[date]['close'] if not vix_data.empty and date in vix_data.index else np.nan,
                        'rsi_on_entry': setup_candle.get('rsi_14', np.nan),
                        'volume_ratio_on_entry': setup_candle.get('volume_ratio', np.nan),
                        'close_to_ema_ratio': (setup_candle['close'] / ema_val - 1) * 100 if pd.notna(ema_val) and ema_val > 0 else np.nan
                    })

                if ef_cfg.get('use_volume_filter') and setup_candle.get('volume_ratio', 0) < ef_cfg['min_volume_ratio']:
                    log_filtered('Volume')
                    continue
                if ef_cfg.get('use_rsi_filter'):
                    rsi = setup_candle.get('rsi_14', 50)
                    if not (ef_cfg['min_rsi'] <= rsi <= ef_cfg['max_rsi']):
                        log_filtered('RSI')
                        continue
                
                ema_val = setup_candle.get(f"ema_{ef_cfg['ema_period']}", np.nan)
                dist_ratio = (setup_candle['close'] / ema_val - 1) * 100 if pd.notna(ema_val) and ema_val > 0 else np.nan
                
                long_ema_dist_ok = not ef_cfg.get('use_ema_distance_filter') or (ef_cfg['min_long_ema_dist_pct'] <= dist_ratio <= ef_cfg['max_long_ema_dist_pct'])
                short_ema_dist_ok = not ef_cfg.get('use_ema_distance_filter') or (ef_cfg['min_short_ema_dist_pct'] <= dist_ratio <= ef_cfg['max_short_ema_dist_pct'])

                long_preceding_ok = (ef_cfg['preceding_candle_long'] == 'any') or (ef_cfg['preceding_candle_long'] == 'red' and preceding_candle['red_candle']) or (ef_cfg['preceding_candle_long'] == 'green' and preceding_candle['green_candle'])
                short_preceding_ok = (ef_cfg['preceding_candle_short'] == 'any') or (ef_cfg['preceding_candle_short'] == 'red' and preceding_candle['red_candle']) or (ef_cfg['preceding_candle_short'] == 'green' and preceding_candle['green_candle'])

                ema_col = f"ema_{ef_cfg['ema_period']}"
                if ef_cfg['use_ema_filter'] and ema_col in setup_candle:
                    is_uptrend = setup_candle['close'] > setup_candle[ema_col]
                    is_downtrend = setup_candle['close'] < setup_candle[ema_col]
                else:
                    is_uptrend, is_downtrend = True, True

                is_valid_setup = (cfg['setup_candle_type'] == 'both') or (cfg['setup_candle_type'] == 'red' and setup_candle['red_candle']) or (cfg['setup_candle_type'] == 'green' and setup_candle['green_candle'])

                if is_valid_setup:
                    watchlist[symbol] = {'trigger_date': next_day}
                    
                    if is_uptrend and long_preceding_ok:
                        if not long_ema_dist_ok:
                            log_filtered('EMA Distance (Long)')
                        else:
                            valid_long = True
                            if setup_candle['green_candle'] and ef_cfg['filter_green_candle_wick']:
                                rng = setup_candle['high'] - setup_candle['low']
                                if rng > 0 and ((setup_candle['high'] - setup_candle['close']) / rng * 100) > ef_cfg['max_green_wick_percent']: valid_long = False
                                elif rng == 0: valid_long = False
                            if valid_long:
                                watchlist[symbol]['long'] = {'direction': 'long', 'trigger_price': setup_candle['high'], 'stop_loss_price': setup_candle['low'], 'setup_candle_type': 'Red' if setup_candle['red_candle'] else 'Green', 'setup_candle_date': date}

                    if cfg['allow_shorting'] and is_downtrend and short_preceding_ok:
                        if not short_ema_dist_ok:
                            log_filtered('EMA Distance (Short)')
                        else:
                            valid_short = True
                            if setup_candle['red_candle'] and ef_cfg['filter_red_candle_wick']:
                                rng = setup_candle['high'] - setup_candle['low']
                                if rng > 0 and ((setup_candle['open'] - setup_candle['low']) / rng * 100) > ef_cfg['max_red_wick_percent']: valid_short = False
                                elif rng == 0: valid_short = False
                            if valid_short:
                               watchlist[symbol]['short'] = {'direction': 'short', 'trigger_price': setup_candle['low'], 'stop_loss_price': setup_candle['high'], 'setup_candle_type': 'Green' if setup_candle['green_candle'] else 'Red', 'setup_candle_date': date}

    trades_df = pd.DataFrame(portfolio['trades'])
    total_trades = len(trades_df)
    if total_trades == 0: return (pd.DataFrame(), {}, [])
    
    equity_df = pd.DataFrame(portfolio['daily_values']).set_index('date')
    cagr, max_drawdown = 0, 0
    if not equity_df.empty and len(equity_df) > 1:
        years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = ((portfolio['equity'] / cfg['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100

    win_rate = (len(trades_df[trades_df['pnl'] > 0]) / total_trades) * 100
    net_pnl = trades_df['pnl'].sum()
    
    long_trades_df = trades_df[trades_df['direction'] == 'long']
    short_trades_df = trades_df[trades_df['direction'] == 'short']
    
    summary = {
        'symbol': symbol, 'total_trades': total_trades, 'win_rate': win_rate, 
        'net_pnl': net_pnl, 'cagr': cagr, 'max_drawdown': max_drawdown,
        'long_trades': len(long_trades_df), 'long_pnl': long_trades_df['pnl'].sum(),
        'short_trades': len(short_trades_df), 'short_pnl': short_trades_df['pnl'].sum()
    }
    return (trades_df, summary, filtered_setups)

# --- Main Orchestration Function ---
def main():
    cfg = config
    print(f"--- Starting Backtest: {cfg['strategy_name']} ---")
    
    if cfg['target_stock']:
        symbols = [cfg['target_stock']]
        print(f"Running in single stock mode for: {cfg['target_stock']}")
    else:
        try:
            symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
            print(f"Running in universe mode for {len(symbols)} stocks.")
        except FileNotFoundError:
            print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return

    print(f"Loading data...")
    all_data = {}
    data_path = os.path.join(cfg['data_folder'], cfg['timeframe'])
    for symbol in symbols:
        file_path = os.path.join(data_path, f"{symbol}_{cfg['timeframe']}_with_indicators.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
            df.rename(columns=lambda x: x.lower(), inplace=True)
            df['red_candle'] = df['close'] < df['open']
            df['green_candle'] = df['close'] > df['open']
            all_data[symbol] = df.loc[cfg['start_date']:cfg['end_date']]
            
    vix_data = pd.DataFrame()
    vix_path = os.path.join(data_path, f"{cfg['vix_symbol']}_{cfg['timeframe']}_with_indicators.csv")
    if os.path.exists(vix_path):
        vix_data = pd.read_csv(vix_path, index_col='datetime', parse_dates=True)
    else:
        print(f"Warning: VIX data not found at {vix_path}")
        
    print("Data loading complete.")

    all_trades_list, all_summaries_list, all_filtered_list = [], [], []
    for i, symbol in enumerate(symbols):
        progress_str = f"Processing {symbol} ({i+1}/{len(symbols)})"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()
        trades_df, summary_dict, filtered_setups = run_backtest_for_symbol(symbol, cfg, all_data, vix_data)
        if not trades_df.empty: all_trades_list.append(trades_df)
        if summary_dict: all_summaries_list.append(summary_dict)
        if filtered_setups: all_filtered_list.extend(filtered_setups)

    print("\n\n--- AGGREGATING RESULTS ---")
    if not all_trades_list: print("No trades were generated across the entire universe."); return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_log_folder = os.path.join(cfg['log_folder'], cfg['strategy_name'], timestamp)
    os.makedirs(strategy_log_folder, exist_ok=True)
    
    final_trades_df = pd.concat(all_trades_list, ignore_index=True)
    final_summary_df = pd.DataFrame(all_summaries_list)
    final_filtered_df = pd.DataFrame(all_filtered_list)

    if cfg['log_options']['log_summary']:
        total_trades, overall_pnl = final_trades_df.shape[0], final_trades_df['pnl'].sum()
        overall_win_rate = (len(final_trades_df[final_trades_df['pnl'] > 0]) / total_trades) * 100 if total_trades > 0 else 0
        
        long_trades = final_trades_df[final_trades_df['direction'] == 'long']
        short_trades = final_trades_df[final_trades_df['direction'] == 'short']
        
        tm_cfg = cfg['trade_management']
        is_trailing_active = tm_cfg.get('use_atr') or tm_cfg.get('use_breakeven') or tm_cfg.get('use_aggressive')
        rr_string = "N/A (Trailing Active)" if is_trailing_active else f"{cfg['risk_reward_ratio']}:1"

        all_stocks_summary_df = final_summary_df.sort_values(by='net_pnl', ascending=False)
        
        def format_config(cfg):
            config_str = "Configuration Settings:\n"
            config_str += "-----------------------\n"
            for key, value in cfg.items():
                if isinstance(value, dict):
                    config_str += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        config_str += f"  - {sub_key}: {sub_value}\n"
                else:
                    config_str += f"{key}: {value}\n"
            config_str += "\n"
            return config_str

        config_details = format_config(cfg)
        
        summary_text = f"""
{config_details}
UNIVERSE BACKTEST SUMMARY: {cfg['strategy_name']}
======================================================
Overall Performance:
--------------------
Total Trades: {total_trades} | Overall Win Rate: {overall_win_rate:.2f}% | Total Net PnL: {overall_pnl:,.2f}
R:R Ratio: {rr_string}

Long vs. Short Performance:
---------------------------
Long Trades: {len(long_trades)} | Win Rate: {(len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0:.2f}% | PnL: {long_trades['pnl'].sum():,.2f}
Short Trades: {len(short_trades)} | Win Rate: {(len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0:.2f}% | PnL: {short_trades['pnl'].sum():,.2f}

Stock-wise Performance Summary:
-------------------------------
{all_stocks_summary_df[['symbol', 'net_pnl', 'total_trades', 'win_rate']].to_string(index=False)}
"""
        
        top_5_winners = final_summary_df.nlargest(5, 'net_pnl')
        top_5_losers = final_summary_df.nsmallest(5, 'net_pnl')
        console_summary_text = f"""
UNIVERSE BACKTEST SUMMARY: {cfg['strategy_name']}
======================================================
Overall Performance:
--------------------
Total Trades: {total_trades} | Overall Win Rate: {overall_win_rate:.2f}% | Total Net PnL: {overall_pnl:,.2f}
R:R Ratio: {rr_string}

Long vs. Short Performance:
---------------------------
Long Trades: {len(long_trades)} | Win Rate: {(len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0:.2f}% | PnL: {long_trades['pnl'].sum():,.2f}
Short Trades: {len(short_trades)} | Win Rate: {(len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0:.2f}% | PnL: {short_trades['pnl'].sum():,.2f}

Top 5 Winners (by PnL):
-----------------------
{top_5_winners.to_string(index=False)}

Top 5 Losers (by PnL):
----------------------
{top_5_losers.to_string(index=False)}
"""
        print(console_summary_text)
        with open(os.path.join(strategy_log_folder, f"{timestamp}_summary.txt"), 'w') as f: f.write(summary_text)
        print(f"Universe summary report saved to '{os.path.join(strategy_log_folder, f'{timestamp}_summary.txt')}'")

    if cfg['log_options']['log_trades']:
        final_trades_df.to_csv(os.path.join(strategy_log_folder, f"{timestamp}_all_trades.csv"), index=False)
        print(f"Aggregated trade log saved to '{os.path.join(strategy_log_folder, f'{timestamp}_all_trades.csv')}'")
    
    if cfg['log_options'].get('log_long_short_trades', False):
        if not long_trades.empty:
            long_trades_filename = os.path.join(strategy_log_folder, f"{timestamp}_long_trades.csv")
            long_trades.to_csv(long_trades_filename, index=False)
            print(f"Long trades log saved to '{long_trades_filename}'")
        if not short_trades.empty:
            short_trades_filename = os.path.join(strategy_log_folder, f"{timestamp}_short_trades.csv")
            short_trades.to_csv(short_trades_filename, index=False)
            print(f"Short trades log saved to '{short_trades_filename}'")

    if cfg['log_options'].get('log_pnl_by_stock', False) and not final_summary_df.empty:
        columns_order = [
            'symbol', 'net_pnl', 'total_trades', 'win_rate', 'cagr', 'max_drawdown',
            'long_trades', 'long_pnl', 'short_trades', 'short_pnl'
        ]
        pnl_by_stock_df = final_summary_df[columns_order].sort_values(by='net_pnl', ascending=False)
        pnl_filename = os.path.join(strategy_log_folder, f"{timestamp}_pnl_by_stock.csv")
        pnl_by_stock_df.to_csv(pnl_filename, index=False)
        print(f"Detailed PnL by stock saved to '{pnl_filename}'")
        
    if cfg['log_options'].get('log_filtered_setups', False) and not final_filtered_df.empty:
        filtered_filename = os.path.join(strategy_log_folder, f"{timestamp}_filtered_setups.csv")
        final_filtered_df.to_csv(filtered_filename, index=False)
        print(f"Filtered setups log saved to '{filtered_filename}'")


if __name__ == "__main__":
    main()
