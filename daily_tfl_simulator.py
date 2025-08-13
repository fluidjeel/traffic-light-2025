# daily_tfl_simulator.py
#
# Description:
# A backtester based on the v5 architecture, enhanced with a dynamic portfolio
# risk system and a VIX Fear Filter to mitigate drawdowns during market panic.
#
# MODIFICATION (Final Version - Two-Stage Exit):
# 1. REVERTED: Entry filters are reset to the most restrictive, highest-quality
#    settings that produced the best Profit Factor.
# 2. NEW FEATURE: Implemented an advanced two-stage exit strategy that can be
#    toggled on or off for direct comparison with the single-target model.
#    - Stage 1: Takes a partial profit at a high-probability first target (1.25R).
#    - Stage 2: Moves the stop to breakeven and manages the remaining position
#      with an ATR trailing stop to capture larger trends.

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
    'start_date': '2023-01-01',
    'end_date': '2025-07-15',
    'nifty_list_csv': 'nifty500.csv',

    # --- Data & Logging ---
    'data_pipeline_config': {
        'use_universal_pipeline': True,
        'universal_processed_folder': os.path.join('data', 'universal_processed'),
        'universal_intraday_folder': os.path.join('data', 'universal_historical_data'),
    },
    'log_folder': 'backtest_logs',
    'strategy_name': 'daily_tfl_simulator',
    'log_options': { 'log_trades': True, 'log_missed': True, 'log_summary': True, 'log_filtered': True, 'log_equity': True, 'log_regime': True },

    # --- Core Daily Strategy Parameters (EOD Scan) ---
    'timeframe': 'daily',
    'use_ema_filter': True, # ADDITIVE CHANGE
    'ema_period': 30,
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200_INDEX',
    'regime_ma_period': 50,
    'secondary_market_regime_filter': {
        'use_secondary_filter': True,
        'ema_period': 20
    },
    'volume_filter': False,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    
    # --- Dynamic Portfolio Risk ---
    'dynamic_portfolio_risk': {
        'use_dynamic_risk': True,
        'healthy_regime_max_risk': 10.0,
        'weakening_regime_max_risk': 5.0
    },

    # --- VIX Fear Filter ---
    'vix_fear_filter': {
        'use_filter': True,
        'spike_threshold_pct': 40.0,
        'sma_period': 10
    },

    # --- Setup Candle Pattern Filter ---
    'setup_candle_filter': {
        'mode': 'Not Bearish',
        'params': {
            'wick_body_ratio_threshold': 2.0,
            'body_range_ratio_threshold': 0.7,
            'doji_body_threshold': 0.05,
        }
    },

    # --- Advanced Trade Management ---
    'trade_management': {
        'stop_loss_mode': 'ATR',
        'atr_period': 14,
        'atr_multiplier': 2.5,
        'use_atr_trailing_stop': True,
        'trailing_stop_atr_multiplier': 1.8,
        
        # --- NEW TOGGLEABLE TWO-STAGE EXIT LOGIC ---
        'use_two_stage_exit': False, # Set to False to revert to single target
        'first_target_R': 1.25,
        'profit_target_multiplier': 5.00, # Used only if use_two_stage_exit is False
    },

    # --- Risk & Position Sizing ---
    'use_dynamic_position_sizing': True,
    'max_portfolio_risk_percent': 10.0,
    'risk_per_trade_percent': 2.0,
    'max_new_positions_per_day': 5,
    
    # --- Intraday Conviction Filters ---
    'use_volume_projection': False,
    'use_enhanced_market_strength': False,

    # --- Slippage & VIX Settings ---
    'use_slippage': True,
    'adaptive_slippage': True,
    'vix_threshold': 25,
    'base_slippage_percent': 0.05,
    'high_vol_slippage_percent': 0.15,
    'vix_symbol': 'INDIAVIX',

    # --- REVERTED TO "ELITE" ENTRY FILTERS ---
    'entry_filters': {
        'filter_prox_52w_high': True,
        'min_prox_52w_high': -10.0,
        'filter_volume_ratio': True,
        'min_volume_ratio': 1.5,
        'filter_low_wick_candle': True,
        'max_wick_percent': 10.0, # Reverted to the stricter 10%
    }
}


# --- HELPER FUNCTIONS ---
def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 1
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def check_candle_patterns(setup_candle, prev_candle, mode, params):
    if mode == 'None': return True
    body_size = abs(setup_candle['close'] - setup_candle['open'])
    candle_range = setup_candle['high'] - setup_candle['low']
    if candle_range == 0: return False
    upper_wick = setup_candle['high'] - max(setup_candle['open'], setup_candle['close'])
    lower_wick = min(setup_candle['open'], setup_candle['close']) - setup_candle['low']
    is_shooting_star = (upper_wick > body_size * params['wick_body_ratio_threshold'])
    is_doji = (body_size / candle_range) < params['doji_body_threshold']
    if mode == 'Not Bearish': return not (is_shooting_star or is_doji)
    is_hammer = (lower_wick > body_size * params['wick_body_ratio_threshold'])
    is_bullish_engulfing = (prev_candle is not None) and (prev_candle['red_candle']) and \
                           (setup_candle['open'] < prev_candle['open']) and (setup_candle['close'] > prev_candle['close'])
    is_solid_green = (body_size / candle_range) > params['body_range_ratio_threshold']
    
    if mode == 'Bullish':
        return is_hammer or is_bullish_engulfing or is_solid_green
    
    return False

def create_enhanced_trade_log(pos, exit_time, exit_price, exit_type, shares_exited):
    base_log = pos.copy()
    base_log.update({'exit_date': exit_time, 'exit_price': exit_price, 'pnl': (exit_price - pos['entry_price']) * shares_exited, 'exit_type': exit_type, 'shares_exited': shares_exited})
    mae_price = pos['lowest_price_since_entry']; mfe_price = pos['highest_price_since_entry']
    if pos['entry_price'] > 0:
        mae_percent = ((pos['entry_price'] - mae_price) / pos['entry_price']) * 100
        mfe_percent = ((mfe_price - pos['entry_price']) / pos['entry_price']) * 100
    else: mae_percent, mfe_percent = 0, 0

    base_log.update({'mae_price': mae_price, 'mae_percent': mae_percent, 'mfe_price': mfe_price, 'mfe_percent': mfe_percent, 'mfe_R': (mfe_price - pos['entry_price']) / (pos['entry_price'] - pos['initial_stop_loss']) if (pos['entry_price'] - pos['initial_stop_loss']) != 0 else np.inf})

    base_log.update({
        'pullback_depth': pos.get('pullback_depth', np.nan),
        'volatility_ratio': pos.get('volatility_ratio', np.nan),
        'prox_52w_high': pos.get('prox_52w_high', np.nan),
        'volume_ratio': pos.get('volume_ratio', np.nan),
        'atr_percent': pos.get('atr_percent', np.nan)
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
                df_i = pd.read_csv(intraday_file, index_col='datetime', parse_dates=True)
                df_i.rename(columns=lambda x: x.lower(), inplace=True)
                intraday_data[symbol] = df_i
        except Exception as e:
            print(f"Warning: Could not load data for {symbol}. Error: {e}")
    print("Data loading complete.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    watchlist = {}
    
    master_dates = daily_data[cfg['regime_index_symbol']].loc[cfg['start_date']:cfg['end_date']].index

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_log_folder = os.path.join(cfg['log_folder'], cfg['strategy_name'], timestamp)
    os.makedirs(strategy_log_folder, exist_ok=True)
    
    print(f"Starting Simulator. Logs will be saved to: {strategy_log_folder}")
    for date in master_dates:
        todays_max_portfolio_risk = cfg['max_portfolio_risk_percent']
        market_uptrend = True
        regime_status = 'Bear Market'
        index_df = daily_data[cfg['regime_index_symbol']]
        
        if cfg['market_regime_filter'] and date in index_df.index:
            long_term_ok = index_df.loc[date]['close'] >= index_df.loc[date][f"ema_{cfg['regime_ma_period']}"]
            if not long_term_ok:
                market_uptrend = False
            else:
                regime_status = 'Healthy'
                if cfg['dynamic_portfolio_risk']['use_dynamic_risk']:
                    secondary_filter_cfg = cfg.get('secondary_market_regime_filter', {})
                    if secondary_filter_cfg.get('use_secondary_filter'):
                        short_ema_period = secondary_filter_cfg.get('ema_period', 20)
                        short_ema_col = f"ema_{short_ema_period}"
                        short_term_ok = index_df.loc[date]['close'] >= index_df.loc[date][short_ema_col]

                        if short_term_ok:
                            todays_max_portfolio_risk = cfg['dynamic_portfolio_risk']['healthy_regime_max_risk']
                        else:
                            todays_max_portfolio_risk = cfg['dynamic_portfolio_risk']['weakening_regime_max_risk']
                            regime_status = 'Weakening'
        
        vix_filter_cfg = cfg.get('vix_fear_filter', {})
        if vix_filter_cfg.get('use_filter'):
            vix_df = daily_data[cfg['vix_symbol']]
            if date in vix_df.index:
                vix_today = vix_df.loc[date]
                sma_col = f"vix_{vix_filter_cfg.get('sma_period', 10)}_sma"
                
                if pd.notna(vix_today.get(sma_col)) and vix_today[sma_col] > 0:
                    spike_pct = ((vix_today['close'] / vix_today[sma_col]) - 1) * 100
                    
                    if spike_pct > vix_filter_cfg.get('spike_threshold_pct', 30.0):
                        market_uptrend = False
                        regime_status = 'High Fear'

        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Regime: {regime_status}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()

        todays_watchlist = watchlist.get(date, {})
        todays_new_positions = 0

        precomputed_data = {}
        for symbol, details in todays_watchlist.items():
            if symbol in intraday_data:
                try:
                    df_intra = intraday_data[symbol].loc[date.date().strftime('%Y-%m-%d')]
                    precomputed_data[symbol] = {'intraday_candles': df_intra}
                except KeyError: continue
        
        if todays_watchlist and precomputed_data:
            try:
                vix_df = daily_data[cfg['vix_symbol']]
                vix_close_t1 = vix_df.loc[:date].iloc[-2]['close']
                today_intraday_candles = intraday_data.get(cfg['regime_index_symbol'], pd.DataFrame()).loc[date.date().strftime('%Y-%m-%d')]
            except (KeyError, IndexError): continue

            if not today_intraday_candles.empty:
                for candle_idx, (candle_time, _) in enumerate(today_intraday_candles.iterrows()):
                    exit_proceeds, to_remove = 0, []
                    tm_cfg = cfg['trade_management'] # BUG FIX: Define tm_cfg at a higher scope
                    for pos_id, pos in list(portfolio['positions'].items()):
                        if pos['symbol'] not in intraday_data or candle_time not in intraday_data[pos['symbol']].index: continue
                        candle = intraday_data[pos['symbol']].loc[candle_time]

                        pos['lowest_price_since_entry'] = min(pos.get('lowest_price_since_entry', pos['entry_price']), candle['low'])
                        pos['highest_price_since_entry'] = max(pos.get('highest_price_since_entry', pos['entry_price']), candle['high'])

                        if tm_cfg.get('use_two_stage_exit'):
                            # --- TWO-STAGE EXIT LOGIC ---
                            if not pos.get('partial_exit', False) and candle['high'] >= pos['first_target']:
                                exit_price = max(pos['first_target'], candle['open'])
                                shares_to_sell = pos['shares'] // 2
                                if shares_to_sell > 0:
                                    trade_log = create_enhanced_trade_log(pos, candle_time, exit_price, 'Partial Profit (Stage 1)', shares_to_sell)
                                    portfolio['trades'].append(trade_log)
                                    exit_proceeds += shares_to_sell * exit_price
                                    pos['shares'] -= shares_to_sell
                                    pos['partial_exit'] = True
                                    pos['stop_loss'] = pos['entry_price'] # Move to breakeven
                        else:
                            # --- SINGLE TARGET EXIT LOGIC (ORIGINAL) ---
                            if candle['high'] >= pos['target']:
                                exit_price = max(pos['target'], candle['open'])
                                trade_log = create_enhanced_trade_log(pos, candle_time, exit_price, 'Profit Target', pos['shares'])
                                portfolio['trades'].append(trade_log)
                                exit_proceeds += pos['shares'] * exit_price
                                to_remove.append(pos_id)
                                continue

                        if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                            exit_type = 'Stop-Loss'
                            if pos.get('partial_exit', False):
                                exit_type = 'Stop-Loss (Runner)'
                            
                            exit_price = min(pos['stop_loss'], candle['open'])
                            trade_log = create_enhanced_trade_log(pos, candle_time, exit_price, exit_type, pos['shares'])
                            portfolio['trades'].append(trade_log)
                            exit_proceeds += pos['shares'] * exit_price; to_remove.append(pos_id)
                    
                    for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
                    portfolio['cash'] += exit_proceeds

                    if todays_new_positions >= cfg['max_new_positions_per_day']: continue
                    
                    for symbol, details in list(todays_watchlist.items()):
                        if symbol not in precomputed_data or candle_time not in precomputed_data[symbol]['intraday_candles'].index or any(p['symbol'] == symbol for p in portfolio['positions'].values()): continue
                        
                        candle = precomputed_data[symbol]['intraday_candles'].loc[candle_time]
                        
                        if candle['high'] >= details['trigger_price']:
                            entry_price_base = max(details['trigger_price'], candle['open'])
                            slippage_pct = cfg['high_vol_slippage_percent'] if cfg['adaptive_slippage'] and vix_close_t1 > cfg['vix_threshold'] else cfg['base_slippage_percent']
                            entry_price = entry_price_base * (1 + (slippage_pct / 100))

                            atr_col = f"atr_{tm_cfg['atr_period']}"
                            daily_df_loc = daily_data[symbol].index.get_loc(date)
                            
                            atr_value = daily_data[symbol].iloc[daily_df_loc - 1][atr_col]
                            stop_loss = entry_price - (atr_value * tm_cfg['atr_multiplier'])
                            
                            if pd.isna(stop_loss) or entry_price <= stop_loss: continue
                            risk_per_share = entry_price - stop_loss
                            
                            capital_for_this_trade = portfolio['cash'] * (cfg['risk_per_trade_percent'] / 100)

                            if cfg['use_dynamic_position_sizing']:
                                active_risk = sum([(p['entry_price'] - p['stop_loss']) * p['shares'] for p in portfolio['positions'].values()])
                                max_total_risk = portfolio['equity'] * (todays_max_portfolio_risk / 100)
                                available_risk_capital = max(0, max_total_risk - active_risk)
                                capital_for_this_trade = min(available_risk_capital, capital_for_this_trade)
                            
                            shares = math.floor(capital_for_this_trade / risk_per_share) if risk_per_share > 0 else 0
                            
                            if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                                portfolio['cash'] -= shares * entry_price
                                todays_new_positions += 1
                                
                                position_data = {
                                    'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price,
                                    'stop_loss': stop_loss, 'initial_stop_loss': stop_loss,
                                    'shares': shares,
                                    'first_target': entry_price + (risk_per_share * tm_cfg['first_target_R']) if tm_cfg.get('use_two_stage_exit') else np.inf,
                                    'target': entry_price + (risk_per_share * tm_cfg['profit_target_multiplier']) if not tm_cfg.get('use_two_stage_exit') else np.inf,
                                    'partial_exit': False,
                                    'setup_id': details['setup_id'],
                                    'lowest_price_since_entry': entry_price, 'highest_price_since_entry': entry_price,
                                    'vix_on_entry': vix_close_t1,
                                    'atr_percent': details.get('atr_percent', np.nan)
                                }
                                portfolio['positions'][f"{symbol}_{candle_time}"] = position_data
                            
                            del todays_watchlist[symbol]

        # --- GENERATE WATCHLIST FOR NEXT DAY ---
        if market_uptrend:
            for symbol in symbols:
                if symbol not in daily_data or date not in daily_data[symbol].index: continue
                df = daily_data[symbol]
                try:
                    loc = df.index.get_loc(date)
                    if loc < 2: continue
                    setup_candle = df.iloc[loc]
                    
                    if cfg.get('use_ema_filter', True):
                        if not (setup_candle['green_candle'] and setup_candle['close'] > setup_candle[f"ema_{cfg['ema_period']}"]):
                            continue
                    elif not setup_candle['green_candle']:
                        continue
                    
                    if not get_consecutive_red_candles(df, loc):
                        continue
                    
                    entry_filters_cfg = cfg.get('entry_filters', {})
                    if entry_filters_cfg.get('filter_low_wick_candle'):
                        candle_range = setup_candle['high'] - setup_candle['low']
                        if candle_range > 0:
                            upper_wick = setup_candle['high'] - max(setup_candle['open'], setup_candle['close'])
                            if (upper_wick / candle_range) * 100 > entry_filters_cfg['max_wick_percent']:
                                continue
                        else:
                            continue

                    if entry_filters_cfg.get('filter_prox_52w_high'):
                        prox_value = setup_candle.get('prox_52w_high', np.nan)
                        if pd.notna(prox_value) and prox_value < entry_filters_cfg.get('min_prox_52w_high'):
                            continue
                    
                    if entry_filters_cfg.get('filter_volume_ratio'):
                        vol_ratio_value = setup_candle.get('volume_ratio', np.nan)
                        if pd.notna(vol_ratio_value) and vol_ratio_value < entry_filters_cfg.get('min_volume_ratio'):
                            continue
                    
                    candle_filter_cfg = cfg.get('setup_candle_filter', {'mode': 'None'})
                    if candle_filter_cfg['mode'] != 'None':
                        prev_candle = df.iloc[loc - 1] if loc > 0 else None
                        if not check_candle_patterns(setup_candle, prev_candle, candle_filter_cfg['mode'], candle_filter_cfg['params']):
                            continue

                    trigger_price = setup_candle['high']
                    setup_id = f"{symbol}_{date.strftime('%Y-%m-%d')}"
                    
                    atr_percent_log = np.nan
                    atr_val_log = setup_candle.get(f"atr_{cfg['trade_management']['atr_period']}", np.nan)
                    if pd.notna(atr_val_log) and setup_candle['close'] > 0:
                        atr_percent_log = (atr_val_log / setup_candle['close']) * 100

                    next_day = date + timedelta(days=1)
                    if next_day in master_dates:
                        # BUG FIX: Use the datetime object as the key
                        if next_day not in watchlist: watchlist[next_day] = {}
                        watchlist[next_day][symbol] = {
                            'trigger_price': trigger_price,
                            'setup_id': setup_id,
                            'atr_value': setup_candle[f"atr_{cfg['trade_management']['atr_period']}"],
                            'atr_percent': atr_percent_log
                        }
                except (KeyError, IndexError): continue

        # --- EOD TRAILING STOP LOGIC ---
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                tm_cfg = cfg['trade_management']
                daily_candle = daily_data[pos['symbol']].loc[date]
                new_stop = pos['stop_loss']
                
                if tm_cfg.get('use_atr_trailing_stop', False):
                    current_atr = daily_candle[f"atr_{tm_cfg['atr_period']}"]
                    
                    if regime_status == 'Weakening':
                        trailing_stop_multiplier = 1.5
                    else:
                        trailing_stop_multiplier = tm_cfg['trailing_stop_atr_multiplier']

                    atr_stop = pos['highest_price_since_entry'] - (current_atr * trailing_stop_multiplier)
                    new_stop = max(new_stop, atr_stop)
                
                pos['stop_loss'] = new_stop
        
        eod_equity = portfolio['cash'] + sum([p['shares'] * daily_data[p['symbol']].loc[date]['close'] for p in portfolio['positions'].values() if date in daily_data.get(p['symbol'], pd.DataFrame()).index])
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    # --- FINALIZE & SAVE RESULTS ---
    print("\n\n--- BACKTEST COMPLETE ---")
    
    log_opts = cfg.get('log_options', {})
    trades_df = pd.DataFrame(portfolio['trades'])
    final_equity = portfolio['equity']
    
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
            
            if cfg['trade_management'].get('use_two_stage_exit'):
                unique_winning_positions = len(trades_df[trades_df['pnl'] > 0]['setup_id'].unique())
                total_positions = len(trades_df['setup_id'].unique())
                win_rate = (unique_winning_positions / total_positions) * 100 if total_positions > 0 else 0
            else:
                total_positions = len(trades_df)
                win_rate = (len(winning_trades) / total_positions) * 100 if total_positions > 0 else 0

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
Total Positions: {total_positions}
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

if __name__ == "__main__":
    run_backtest(config)
