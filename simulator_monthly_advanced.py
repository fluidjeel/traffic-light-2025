# simulator_monthly_advanced.py
#
# Description:
# This is a new, state-of-the-art, bias-free backtesting simulator for a
# Monthly Pullback Strategy, inspired by the HTF (weekly) framework.
#
# MODIFICATION (v3.0 - FINAL CALCULATION FIX):
# 1. FIXED: A critical flaw where the previous fix for position sizing was
#    incomplete. The logic now correctly uses the equity at the start of the
#    day ('equity_at_sod') for BOTH position sizing modes ('use_dynamic_position_sizing'
#    set to True or False), guaranteeing that unrealized profits are never
#    used as leverage. This ensures robust and realistic results.
#
# MODIFICATION (v2.8 - Custom Risk Model):
# 1. ADDED: A 'profit_target_mode' to the config for custom risk modeling.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time, timedelta
import sys
import numpy as np

# --- CONFIGURATION FOR THE MONTHLY ADVANCED SIMULATOR ---
config = {
    # --- General Backtest Parameters ---
    'initial_capital': 1000000,
    'nifty_list_csv': 'nifty200.csv',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',

    # --- Data & Logging Paths ---
    'data_folder_base': 'data/processed',
    'intraday_data_folder': 'historical_data_15min',
    'log_folder': 'backtest_logs',
    'strategy_name': 'simulator_monthly_advanced',

    # --- Enhanced Logging Options ---
    'log_options': { 'log_trades': True, 'log_missed': True, 'log_summary': True, 'log_filtered': True },

    # --- Core Monthly Strategy Parameters (Scout) ---
    'timeframe': 'monthly',
    'use_ema_filter': False,
    'ema_period_monthly': 10,
    'use_monthly_volume_filter': False,
    'volume_ma_period_monthly': 12,
    'volume_multiplier_monthly': 1.2,
    
    # --- Stop-Loss Configuration ---
    'stop_loss_mode': 'PERCENT', # Options: 'ATR' or 'PERCENT'
    'fixed_stop_loss_percent': 0.09, # 9% stop-loss as per MAE analysis
    'atr_period_monthly': 6,
    'atr_multiplier_stop': 1.2,

    # --- Adaptive Execution Window (Sniper) ---
    'execution_window_base_days': 15,
    'execution_window_vix_extension': 7,
    'execution_window_skip_days': 2,

    # --- Portfolio & Risk Management ---
    'use_dynamic_position_sizing': True,
    'risk_per_trade_percent': 2.0,
    'max_portfolio_risk_percent': 15.0,
    'max_new_positions_per_day': 6,

    # --- Conviction Engine (Sniper Filters) ---
    'use_volume_projection': False,
    'volume_ma_period_daily': 20,
    'volume_multiplier_daily': 1.5,
    'volume_projection_thresholds': { dt_time(10, 0): 0.20, dt_time(11, 30): 0.45, dt_time(13, 0): 0.65, dt_time(14, 0): 0.85 },
    'use_vix_adaptive_filters': False,
    'vix_symbol': 'INDIAVIX',
    'vix_high_threshold': 25,
    'market_strength_index': 'NIFTY200_INDEX',
    'market_strength_threshold_calm': -0.25,
    'market_strength_threshold_high': -0.75,
    'base_slippage_percent': 0.10,
    'high_vol_slippage_percent': 0.20,
    'use_intraday_rs_filter': False,

    # --- Trade Management & Profit Taking ---
    'profit_target_mode': 'ATR_BASED', # Options: 'ACTUAL_RISK' or 'ATR_BASED'
    'use_partial_profit_leg': False,
    'use_dynamic_profit_target': True,
    'profit_target_rr_calm': 2.0,
    'profit_target_rr_high': 1.5,
    'use_aggressive_breakeven': True,
    'breakeven_buffer_points': 0.05,
}

# --- HELPER FUNCTIONS ---
def get_consecutive_red_candles(df, current_loc, min_candles=1):
    red_candles = []
    i = current_loc - 1
    while i >= 0 and (df.iloc[i]['close'] < df.iloc[i]['open']):
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles if len(red_candles) >= min_candles else []

def check_volume_projection(candle_time, cumulative_volume, target_daily_volume, thresholds):
    current_time = candle_time.time()
    applicable_threshold = 0
    for threshold_time, threshold_pct in sorted(thresholds.items()):
        if current_time >= threshold_time:
            applicable_threshold = threshold_pct
        else:
            break
    if applicable_threshold == 0: return False, 0, 0
    required_volume = target_daily_volume * applicable_threshold
    return cumulative_volume >= required_volume, cumulative_volume, required_volume

def simulate_trade_outcome(symbol, entry_date, entry_price, stop_loss, daily_data, cfg):
    df = daily_data[symbol]
    if pd.isna(stop_loss): return pd.NaT, np.nan
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
    total_pnl = (partial_exit_pnl * 0.5) + (final_pnl * 0.5) if (cfg['use_partial_profit_leg'] and leg1_sold) else final_pnl
    return exit_date, total_pnl

# --- MAIN SIMULATOR CLASS ---
class MonthlyAdvancedSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
        self.daily_data, self.intraday_data, self.monthly_data = {}, {}, {}
        self.target_list = {}
        self.tradeable_symbols = []
        self.all_setups_log = []
        self.filtered_log = []

    def load_data(self):
        print("Loading all data into memory...")
        try:
            self.tradeable_symbols = pd.read_csv(self.cfg['nifty_list_csv'])['Symbol'].tolist()
        except FileNotFoundError:
            print(f"FATAL ERROR: Symbols file not found. Exiting."); sys.exit()

        symbols_to_load = self.tradeable_symbols + [self.cfg['market_strength_index'], self.cfg['vix_symbol']]
        for symbol in symbols_to_load:
            try:
                for tf, data_map in [('daily', self.daily_data), ('monthly', self.monthly_data)]:
                    file = os.path.join(self.cfg['data_folder_base'], tf, f"{symbol}_{tf}_with_indicators.csv")
                    if os.path.exists(file): data_map[symbol] = pd.read_csv(file, index_col='datetime', parse_dates=True)
                intraday_file = os.path.join(self.cfg['intraday_data_folder'], f"{symbol}_15min.csv")
                if "NIFTY200_INDEX" in symbol: intraday_file = os.path.join(self.cfg['intraday_data_folder'], "NIFTY_200_15min.csv")
                if os.path.exists(intraday_file): self.intraday_data[symbol] = pd.read_csv(intraday_file, index_col='datetime', parse_dates=True)
            except Exception as e:
                print(f"Warning: Could not load all data for {symbol}. Error: {e}")
        print("Data loading complete.")

    def run_simulation(self):
        print("---- RUNNING VERSION 3.0 WITH CALCULATION FIX ----") # Add this line
        if not self.daily_data or self.cfg['market_strength_index'] not in self.daily_data:
            print("Error: Market index data not loaded."); return
        if not self.daily_data or self.cfg['market_strength_index'] not in self.daily_data:
            print("Error: Market index data not loaded."); return
        master_dates = self.daily_data[self.cfg['market_strength_index']].loc[self.cfg['start_date']:self.cfg['end_date']].index
        
        for date in master_dates:
            progress_str = f"Processing {date.date()} | Equity: {self.portfolio['equity']:,.0f} | Positions: {len(self.portfolio['positions'])}"
            sys.stdout.write(f"\r{progress_str.ljust(120)}"); sys.stdout.flush()
            if date.is_month_end: self.scout_for_setups(date)
            self.sniper_monitor_and_execute(date)
            self.manage_eod_portfolio(date)
        self.generate_report()

    def scout_for_setups(self, month_end_date):
        for symbol in self.tradeable_symbols:
            if symbol not in self.monthly_data or symbol not in self.daily_data: continue
            df_monthly, df_daily = self.monthly_data[symbol], self.daily_data[symbol]
            if df_monthly.empty: continue
            try:
                loc_monthly_indexer = df_monthly.index.get_indexer([month_end_date], method='ffill')
                if not loc_monthly_indexer.size or loc_monthly_indexer[0] == -1: continue
                loc_monthly = loc_monthly_indexer[0]
                setup_candle = df_monthly.iloc[loc_monthly]

                is_green = setup_candle['close'] > setup_candle['open']
                
                above_ema = True
                if self.cfg.get('use_ema_filter', True):
                    above_ema = setup_candle['close'] > setup_candle.get(f"ema_{self.cfg['ema_period_monthly']}", np.inf)
                
                volume_ok = True
                if self.cfg.get('use_monthly_volume_filter', False):
                    volume_ok = setup_candle['volume'] > (setup_candle[f"volume_{self.cfg['volume_ma_period_monthly']}_sma"] * self.cfg['volume_multiplier_monthly'])
                
                if not (is_green and above_ema and volume_ok): continue
                
                red_candles = get_consecutive_red_candles(df_monthly, loc_monthly, min_candles=1)
                if not red_candles: continue

                trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])
                if month_end_date not in df_daily.index: continue
                
                vol_sma = df_daily.loc[month_end_date, f"volume_{self.cfg['volume_ma_period_daily']}_sma"]
                target_daily_volume = vol_sma * self.cfg['volume_multiplier_daily']
                monthly_atr = setup_candle[f"atr_{self.cfg['atr_period_monthly']}"]
                if pd.isna(target_daily_volume) or pd.isna(monthly_atr): continue

                setup_id = f"{symbol}_{setup_candle.name.strftime('%Y-%m-%d')}"
                self.all_setups_log.append({'setup_id': setup_id, 'symbol': symbol, 'setup_date': setup_candle.name, 'trigger_price': trigger_price, 'monthly_atr': monthly_atr, 'status': 'IDENTIFIED'})

                vix_close = self.daily_data[self.cfg['vix_symbol']].loc[month_end_date]['close']
                
                start_next_month = month_end_date + pd.offsets.MonthBegin(1)
                end_next_month = month_end_date + pd.offsets.MonthEnd(1)
                all_trading_days = self.daily_data[self.cfg['market_strength_index']].loc[start_next_month:end_next_month].index
                
                window_days = self.cfg['execution_window_base_days']
                if vix_close > self.cfg['vix_high_threshold']:
                    window_days += self.cfg['execution_window_vix_extension']
                execution_window_days = all_trading_days[self.cfg['execution_window_skip_days']:window_days]

                for day in execution_window_days:
                    day_str = day.strftime('%Y-%m-%d')
                    if day_str not in self.target_list: self.target_list[day_str] = {}
                    self.target_list[day_str][symbol] = {'trigger_price': trigger_price, 'target_daily_volume': target_daily_volume, 'monthly_atr': monthly_atr, 'setup_id': setup_id}
            except Exception: continue

    def sniper_monitor_and_execute(self, date):
        date_str = date.strftime('%Y-%m-%d')
        todays_watchlist = self.target_list.get(date_str, {})
        if not todays_watchlist: return
        
        equity_at_sod = self.portfolio['equity']

        try:
            mkt_idx_intra = self.intraday_data[self.cfg['market_strength_index']].loc[date_str]
            if mkt_idx_intra.empty: return
            mkt_open = mkt_idx_intra.iloc[0]['open']
            vix_df = self.daily_data[self.cfg['vix_symbol']]
            current_date_pos_indexer = vix_df.index.get_indexer([date], method='ffill')
            if not current_date_pos_indexer.size or current_date_pos_indexer[0] < 1: return
            prev_day_loc = current_date_pos_indexer[0] - 1
            vix_close = vix_df.iloc[prev_day_loc]['close']
            if isinstance(vix_close, pd.Series): vix_close = vix_close.item()
        except (KeyError, IndexError): return

        todays_new_positions = 0
        for candle_time, mkt_candle in mkt_idx_intra.iterrows():
            self.manage_intraday_exits(candle_time)
            if todays_new_positions >= self.cfg['max_new_positions_per_day']: continue
            for symbol, details in list(todays_watchlist.items()):
                if any(p['symbol'] == symbol for p in self.portfolio['positions'].values()):
                    if symbol in todays_watchlist: del todays_watchlist[symbol]
                    continue
                try:
                    stock_intra_df = self.intraday_data[symbol].loc[date_str]
                    stock_candle = stock_intra_df.loc[candle_time]
                    stock_open = stock_intra_df.iloc[0]['open']
                except (KeyError, IndexError): continue

                if stock_candle['high'] >= details['trigger_price']:
                    log_template = {'symbol': symbol, 'timestamp': candle_time, 'trigger_price': details['trigger_price'], 'setup_id': details['setup_id']}
                    
                    if self.cfg['use_volume_projection']:
                        cum_vol = stock_intra_df.loc[:candle_time, 'volume'].sum()
                        vol_ok, actual_vol, req_vol = check_volume_projection(candle_time, cum_vol, details['target_daily_volume'], self.cfg['volume_projection_thresholds'])
                        if not vol_ok:
                            self.filtered_log.append({**log_template, 'filter_type': 'Volume Projection', 'actual': f"{actual_vol:,.0f}", 'expected': f"{req_vol:,.0f}"})
                            continue
                    if self.cfg['use_vix_adaptive_filters']:
                        mkt_strength = (mkt_candle['close'] / mkt_open - 1) * 100
                        threshold = self.cfg['market_strength_threshold_high'] if vix_close > self.cfg['vix_high_threshold'] else self.cfg['market_strength_threshold_calm']
                        if mkt_strength < threshold:
                            self.filtered_log.append({**log_template, 'filter_type': 'Market Strength', 'actual': f"{mkt_strength:.2f}%", 'expected': f">{threshold:.2f}%"})
                            continue
                    if self.cfg['use_intraday_rs_filter']:
                        stock_rs = (stock_candle['close'] / stock_open - 1) * 100
                        if stock_rs < mkt_strength:
                            self.filtered_log.append({**log_template, 'filter_type': 'Intraday RS', 'actual': f"{stock_rs:.2f}%", 'expected': f">{mkt_strength:.2f}%"})
                            continue
                    
                    self.execute_entry(symbol, details, date, candle_time, vix_close, equity_at_sod)
                    todays_new_positions += 1
                    if symbol in todays_watchlist: del todays_watchlist[symbol]
    
    def execute_entry(self, symbol, details, date, candle_time, vix_close, equity_at_sod):
        entry_price_base = details['trigger_price']
        slippage_pct = self.cfg['base_slippage_percent'] if vix_close <= self.cfg['vix_high_threshold'] else self.cfg['high_vol_slippage_percent']
        entry_price = entry_price_base * (1 + slippage_pct / 100)
        
        if self.cfg['stop_loss_mode'] == 'ATR':
            monthly_atr = details['monthly_atr']
            stop_loss_actual = entry_price - (monthly_atr * self.cfg['atr_multiplier_stop'])
        elif self.cfg['stop_loss_mode'] == 'PERCENT':
            stop_loss_actual = entry_price * (1 - self.cfg['fixed_stop_loss_percent'])
        else:
            return 
        
        risk_per_share_actual = entry_price - stop_loss_actual
        if pd.isna(risk_per_share_actual) or risk_per_share_actual <= 0: return

        if self.cfg['profit_target_mode'] == 'ATR_BASED':
            monthly_atr = details['monthly_atr']
            risk_per_share_for_target = monthly_atr * self.cfg['atr_multiplier_stop']
        else:
            risk_per_share_for_target = risk_per_share_actual

        profit_target_rr = self.cfg['profit_target_rr_calm'] if vix_close <= self.cfg['vix_high_threshold'] else self.cfg['profit_target_rr_high']
        target_price = entry_price + (risk_per_share_for_target * profit_target_rr)
        
        # --- CALCULATION FIX: Use equity_at_sod for both sizing modes ---
        if self.cfg['use_dynamic_position_sizing']:
            active_risk = sum([(p['entry_price'] - p['stop_loss']) * p['shares'] for p in self.portfolio['positions'].values()])
            max_portfolio_risk_capital = equity_at_sod * (self.cfg['max_portfolio_risk_percent'] / 100)
            available_risk_capital = max(0, max_portfolio_risk_capital - active_risk)
            capital_to_risk = min((equity_at_sod * (self.cfg['risk_per_trade_percent'] / 100)), available_risk_capital)
        else:
            capital_to_risk = (equity_at_sod * (self.cfg['risk_per_trade_percent'] / 100))
        
        shares = math.floor(capital_to_risk / risk_per_share_actual) if capital_to_risk > 0 else 0

        setup_log_entry = next((item for item in self.all_setups_log if item['setup_id'] == details['setup_id']), None)
        if setup_log_entry:
            setup_log_entry['trigger_date'] = date
            if shares > 0 and (shares * entry_price) <= self.portfolio['cash']:
                setup_log_entry['status'] = 'FILLED'
                self.portfolio['cash'] -= shares * entry_price
                self.portfolio['positions'][f"{symbol}_{candle_time}"] = {
                    'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price,
                    'stop_loss': stop_loss_actual, 'shares': shares, 'target': target_price,
                    'partial_exit': False, 'initial_shares': shares, 'setup_id': details['setup_id'],
                    'lowest_price_since_entry': entry_price
                }
            elif shares > 0:
                setup_log_entry['status'] = 'MISSED_CAPITAL'
            else:
                setup_log_entry['status'] = 'FILTERED_RISK'

    def manage_intraday_exits(self, candle_time):
        exit_proceeds, to_remove = 0, []
        for pos_id, pos in list(self.portfolio['positions'].items()):
            try:
                candle = self.intraday_data[pos['symbol']].loc[candle_time]
                
                pos['lowest_price_since_entry'] = min(pos.get('lowest_price_since_entry', pos['entry_price']), candle['low'])

                if self.cfg['use_partial_profit_leg'] and not pos.get('partial_exit', False) and candle['high'] >= pos['target']:
                    shares_to_sell = pos['shares'] // 2
                    exit_price = pos['target']
                    exit_proceeds += shares_to_sell * exit_price
                    
                    mae_price = pos['lowest_price_since_entry']
                    mae_percent = ((pos['entry_price'] - mae_price) / pos['entry_price']) * 100
                    trade_log = pos.copy(); trade_log.update({'exit_date': candle_time, 'exit_price': exit_price, 'pnl': (exit_price - pos['entry_price']) * shares_to_sell, 'exit_type': 'Partial Profit', 'mae_price': mae_price, 'mae_percent': mae_percent})
                    self.portfolio['trades'].append(trade_log)
                    
                    pos.update({'shares': pos['shares'] - shares_to_sell, 'partial_exit': True, 'stop_loss': pos['entry_price']})

                if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    exit_proceeds += pos['shares'] * exit_price
                    
                    mae_price = pos['lowest_price_since_entry']
                    mae_percent = ((pos['entry_price'] - mae_price) / pos['entry_price']) * 100
                    trade_log = pos.copy(); trade_log.update({'exit_date': candle_time, 'exit_price': exit_price, 'pnl': (exit_price - pos['entry_price']) * pos['shares'], 'exit_type': 'Stop-Loss', 'mae_price': mae_price, 'mae_percent': mae_percent})
                    self.portfolio['trades'].append(trade_log)

                    to_remove.append(pos_id)
            except (KeyError, IndexError): continue
        for pos_id in to_remove: self.portfolio['positions'].pop(pos_id, None)
        self.portfolio['cash'] += exit_proceeds

    def manage_eod_portfolio(self, date):
        for pos in self.portfolio['positions'].values():
            if date in self.daily_data.get(pos['symbol'], pd.DataFrame()).index:
                daily_candle = self.daily_data[pos['symbol']].loc[date]
                new_stop = pos['stop_loss']
                if daily_candle['close'] > pos['entry_price']:
                    if self.cfg['use_aggressive_breakeven'] and not pos.get('partial_exit', False):
                        new_stop = max(new_stop, pos['entry_price'] + self.cfg['breakeven_buffer_points'])
                    if daily_candle['close'] > daily_candle['open']:
                        new_stop = max(new_stop, daily_candle['low'])
                pos['stop_loss'] = new_stop
        eod_equity = self.portfolio['cash'] + sum([p['shares'] * self.daily_data[p['symbol']].loc[date]['close'] for p in self.portfolio['positions'].values() if date in self.daily_data.get(p['symbol'], pd.DataFrame()).index])
        self.portfolio['equity'] = eod_equity
        self.portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    def generate_report(self):
        print("\n\n--- MONTHLY SIMULATOR BACKTEST COMPLETE ---")
        final_equity = self.portfolio['equity']
        equity_df = pd.DataFrame(self.portfolio['daily_values']).set_index('date')
        cagr, max_drawdown = 0, 0
        if not equity_df.empty:
            years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
            cagr = ((final_equity / self.cfg['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
            peak = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - peak) / peak
            max_drawdown = abs(drawdown.min()) * 100
        trades_df = pd.DataFrame(self.portfolio['trades'])
        total_trades, win_rate, profit_factor = 0, 0, 0
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            total_trades = len(trades_df)
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            gross_profit = winning_trades['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        summary_content = f"""Final Equity: {final_equity:,.2f}, CAGR: {cagr:.2f}%, Max Drawdown: {max_drawdown:.2f}%, Profit Factor: {profit_factor:.2f}, Win Rate: {win_rate:.2f}%, Total Trades: {total_trades}"""
        print(summary_content)

        print("\n--- Generating Enhanced Log Files ---")
        log_opts = self.cfg.get('log_options', {})
        strategy_log_folder = os.path.join(self.cfg['log_folder'], self.cfg['strategy_name'])
        os.makedirs(strategy_log_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if log_opts.get('log_summary', True):
            params_str = "INPUT PARAMETERS:\n-----------------\n"
            for key, value in self.cfg.items():
                if isinstance(value, dict):
                    params_str += f"{key.replace('_', ' ').title()}:\n"
                    for k, v in value.items(): params_str += f"  - {k}: {v}\n"
                else: params_str += f"{key.replace('_', ' ').title()}: {value}\n"
            
            enhanced_summary_content = f"BACKTEST SUMMARY REPORT ({self.cfg['strategy_name'].upper()})\n===================================================================\n{params_str}\n{summary_content}"
            summary_filename = os.path.join(strategy_log_folder, f"{timestamp}_summary.txt")
            with open(summary_filename, 'w') as f: f.write(enhanced_summary_content)
            print(f"Enhanced summary report saved to '{summary_filename}'")

        if log_opts.get('log_trades', True) and not trades_df.empty:
            trades_filename = os.path.join(strategy_log_folder, f"{timestamp}_trade_details.csv")
            trades_df.to_csv(trades_filename, index=False)
            print(f"Trade details saved to '{trades_filename}'")

        if log_opts.get('log_missed', True) and self.all_setups_log:
            all_setups_df = pd.DataFrame(self.all_setups_log)
            missed_trades_df = all_setups_df[all_setups_df['status'].isin(['MISSED_CAPITAL', 'FILTERED_RISK'])].copy()
            if not missed_trades_df.empty:
                missed_trades_df['hypothetical_exit_date'] = pd.NaT
                missed_trades_df['hypothetical_pnl'] = np.nan
                for index, row in missed_trades_df.iterrows():
                    if row['symbol'] in self.daily_data and 'trigger_date' in row and pd.notna(row['trigger_date']):
                        hypothetical_stop_loss = np.nan
                        if 'monthly_atr' in row and pd.notna(row['monthly_atr']):
                            hypothetical_stop_loss = row['trigger_price'] - (row['monthly_atr'] * self.cfg['atr_multiplier_stop'])
                        
                        exit_date, pnl = simulate_trade_outcome(row['symbol'], row['trigger_date'], row['trigger_price'], hypothetical_stop_loss, self.daily_data, self.cfg)
                        missed_trades_df.loc[index, 'hypothetical_exit_date'] = exit_date
                        missed_trades_df.loc[index, 'hypothetical_pnl'] = pnl
                missed_filename = os.path.join(strategy_log_folder, f"{timestamp}_missed_trades.csv")
                missed_trades_df.to_csv(missed_filename, index=False)
                print(f"Missed trades log saved to '{missed_filename}'")
        
        if log_opts.get('log_filtered', True) and self.filtered_log:
            filtered_df = pd.DataFrame(self.filtered_log)
            filtered_filename = os.path.join(strategy_log_folder, f"{timestamp}_filtered.csv")
            filtered_df.to_csv(filtered_filename, index=False)
            print(f"Filtered setups log saved to '{filtered_filename}'")

if __name__ == '__main__':
    simulator = MonthlyAdvancedSimulator(config)
    simulator.load_data()
    simulator.run_simulation()
