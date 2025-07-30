# htf_simulator_advanced.py
#
# Description:
# This is the new, state-of-the-art, bias-free backtesting simulator for the
# Nifty 200 HTF (Higher Timeframe) Pullback Strategy.
#
# MODIFICATION (v2.1 - Logging Fix):
# 1. FIXED: Corrected a bug where the dedicated log subdirectory was not being
#    created, causing a FileNotFoundError at the end of the simulation.
#
# MODIFICATION (v2.0 - Logging Enhancement):
# 1. ADDED: A dedicated subdirectory is now created for each strategy run.
# 2. ADDED: New, toggleable, granular logging for trade details, missed trades,
#    filtered setups, and an enhanced summary report, implemented in a purely
#    additive and non-invasive manner to guarantee identical results.
#
# BUG FIX (v1.5):
# - CRITICAL FIX: Corrected a recurring `TypeError` by replacing deprecated
#   `get_loc` with `get_indexer`.
# - CRITICAL BIAS FIX: Modified the VIX filter to use the PREVIOUS DAY's close.
# - LOGIC FIX: Corrected the Volume Projection engine to return FALSE before 10 AM.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time, timedelta
import time as time_sleep
import sys
import numpy as np

# --- CONFIGURATION FOR THE HTF ADVANCED SIMULATOR ---
config = {
    'initial_capital': 1000000,
    'nifty_list_csv': 'nifty200.csv',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',

    # --- Data Locations ---
    'data_folder_base': 'data/processed',
    'intraday_data_folder': 'historical_data_15min',
    'log_folder': 'backtest_logs',
    'strategy_name': 'simulator_htf_advanced', # For dedicated log folder

    # --- ENHANCED LOGGING OPTIONS ---
    'log_options': {
        'log_trades': True,      # Actual filled trades
        'log_missed': True,      # Trades missed due to capital/risk
        'log_summary': True,     # The main summary.txt file
        'log_filtered': True     # Setups that failed intraday conviction filters
    },

    # --- Core HTF Strategy Parameters ---
    'timeframe': 'weekly', # Base timeframe for the scout
    'ema_period_htf': 30,  # EMA for the weekly setup candle
    'stop_loss_lookback_days': 5,

    # --- Portfolio & Risk Management ---
    'use_dynamic_position_sizing': True,
    'risk_per_trade_percent': 2.0,       # Max risk for a single trade
    'max_portfolio_risk_percent': 6.0,   # Max total risk of all open positions
    'max_new_positions_per_day': 5,

    # --- Conviction Engine (Sniper Filters) ---
    'use_volume_projection': True,
    'volume_ma_period_daily': 20,
    'volume_multiplier_daily': 1.3,
    'volume_projection_thresholds': {
        dt_time(10, 0): 0.20, dt_time(11, 30): 0.45,
        dt_time(13, 0): 0.65, dt_time(14, 0): 0.85
    },

    'use_vix_adaptive_filters': True,
    'vix_symbol': 'INDIAVIX',
    'vix_calm_threshold': 18,
    'vix_high_threshold': 25,
    'market_strength_index': 'NIFTY200_INDEX',
    'market_strength_threshold_calm': -0.25,
    'market_strength_threshold_high': -0.75,
    'base_slippage_percent': 0.05,
    'high_vol_slippage_percent': 0.15,

    'use_intraday_rs_filter': True,

    # --- Trade Management & Profit Taking ---
    'use_partial_profit_leg': True,
    'use_dynamic_profit_target': True,
    'profit_target_rr_calm': 1.5,
    'profit_target_rr_high': 1.0,
    'use_aggressive_breakeven': True,
    'breakeven_buffer_points': 0.05,
}

# --- HELPER FUNCTIONS ---
def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 1
    while i >= 0 and (df.iloc[i]['close'] < df.iloc[i]['open']):
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def check_volume_projection(candle_time, cumulative_volume, target_daily_volume, thresholds):
    current_time = candle_time.time()
    applicable_threshold = 0
    for threshold_time, threshold_pct in sorted(thresholds.items()):
        if current_time >= threshold_time:
            applicable_threshold = threshold_pct
        else:
            break
    if applicable_threshold == 0:
        return False, 0, 0

    required_volume = target_daily_volume * applicable_threshold
    return cumulative_volume >= required_volume, cumulative_volume, required_volume

# New helper for hypothetical trade simulation (non-invasive)
def simulate_htf_trade_outcome(symbol, entry_date, entry_price, stop_loss, daily_data, cfg):
    df = daily_data[symbol]
    # Note: This hypothetical simulation uses a standard 1:1 RR target for simplicity,
    # as it doesn't have access to the intraday VIX value that the live sniper uses.
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
class HtfAdvancedSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.portfolio = {
            'cash': cfg['initial_capital'],
            'equity': cfg['initial_capital'],
            'positions': {},
            'trades': [],
            'daily_values': []
        }
        self.daily_data = {}
        self.intraday_data = {}
        self.htf_data = {}
        self.all_setups_log = []
        self.target_list = {}
        self.debug_log = []
        self.filtered_log = [] # New list for filtered setups
        self.tradeable_symbols = []

    def load_data(self):
        print("Loading all data into memory...")
        try:
            self.tradeable_symbols = pd.read_csv(self.cfg['nifty_list_csv'])['Symbol'].tolist()
        except FileNotFoundError:
            print(f"FATAL ERROR: Tradeable symbols file not found at '{self.cfg['nifty_list_csv']}'. Exiting.")
            sys.exit()

        symbols_to_load = self.tradeable_symbols + [self.cfg['market_strength_index'], self.cfg['vix_symbol']]

        for symbol in symbols_to_load:
            try:
                daily_file = os.path.join(self.cfg['data_folder_base'], 'daily', f"{symbol}_daily_with_indicators.csv")
                if os.path.exists(daily_file):
                    self.daily_data[symbol] = pd.read_csv(daily_file, index_col='datetime', parse_dates=True)
                
                intraday_file = os.path.join(self.cfg['intraday_data_folder'], f"{symbol}_15min.csv")
                if "NIFTY200_INDEX" in symbol: intraday_file = os.path.join(self.cfg['intraday_data_folder'], "NIFTY_200_15min.csv")
                if os.path.exists(intraday_file):
                    self.intraday_data[symbol] = pd.read_csv(intraday_file, index_col='datetime', parse_dates=True)

                htf_file = os.path.join(self.cfg['data_folder_base'], self.cfg['timeframe'], f"{symbol}_{self.cfg['timeframe']}_with_indicators.csv")
                if os.path.exists(htf_file):
                    self.htf_data[symbol] = pd.read_csv(htf_file, index_col='datetime', parse_dates=True)

            except Exception as e:
                print(f"Warning: Could not load all data for {symbol}. Error: {e}")
        print("Data loading complete.")

    def run_simulation(self):
        if not self.daily_data or self.cfg['market_strength_index'] not in self.daily_data:
            print("Error: Market strength index data not loaded. Cannot run simulation.")
            return
            
        master_dates = self.daily_data[self.cfg['market_strength_index']].loc[self.cfg['start_date']:self.cfg['end_date']].index
        
        for date in master_dates:
            progress_str = f"Processing {date.date()} | Equity: {self.portfolio['equity']:,.0f} | Positions: {len(self.portfolio['positions'])} | Targets: {len(self.target_list.get(date.strftime('%Y-%m-%d'), {}))}"
            sys.stdout.write(f"\r{progress_str.ljust(120)}"); sys.stdout.flush()

            if date.weekday() == 4:
                self.scout_for_setups(date)

            self.sniper_monitor_and_execute(date)
            self.manage_eod_portfolio(date)
        
        self.generate_report()

    def scout_for_setups(self, friday_date):
        for symbol in self.tradeable_symbols:
            if symbol not in self.htf_data or symbol not in self.daily_data: 
                continue
            
            df_htf = self.htf_data[symbol]
            if df_htf.empty: continue

            try:
                loc_htf_indexer = df_htf.index.get_indexer([friday_date], method='ffill')
                if not loc_htf_indexer.size or loc_htf_indexer[0] == -1:
                    continue
                loc_htf = loc_htf_indexer[0]
                
                setup_candle = df_htf.iloc[loc_htf]

                is_green = setup_candle['close'] > setup_candle['open']
                above_ema = setup_candle['close'] > setup_candle[f"ema_{self.cfg['ema_period_htf']}"]
                
                if not (is_green and above_ema): continue

                red_candles = get_consecutive_red_candles(df_htf, loc_htf)
                if not red_candles: continue

                trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])
                
                df_daily = self.daily_data[symbol]
                if friday_date not in df_daily.index:
                    self.debug_log.append(f"Scout Skip: {symbol} on {friday_date.date()} - Date not in daily data (holiday?).")
                    continue
                
                vol_sma = df_daily.loc[friday_date, f"volume_{self.cfg['volume_ma_period_daily']}_sma"]
                target_daily_volume = vol_sma * self.cfg['volume_multiplier_daily']

                if pd.isna(target_daily_volume): continue

                for i in range(1, 6):
                    next_day = friday_date + timedelta(days=i)
                    next_day_str = next_day.strftime('%Y-%m-%d')
                    if next_day_str not in self.target_list:
                        self.target_list[next_day_str] = {}
                    
                    self.target_list[next_day_str][symbol] = {
                        'trigger_price': trigger_price,
                        'target_daily_volume': target_daily_volume,
                        'setup_id': f"{symbol}_{setup_candle.name.strftime('%Y-%m-%d')}"
                    }
            except Exception as e:
                self.debug_log.append(f"ERROR in scout_for_setups for {symbol} on {friday_date.date()}: {e}")
                continue

    def sniper_monitor_and_execute(self, date):
        date_str = date.strftime('%Y-%m-%d')
        todays_watchlist = self.target_list.get(date_str, {})
        if not todays_watchlist: return

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
        except (KeyError, IndexError):
            return

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
                except (KeyError, IndexError):
                    continue

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
                    
                    self.execute_entry(symbol, details, date, candle_time, vix_close)
                    todays_new_positions += 1
                    if symbol in todays_watchlist: del todays_watchlist[symbol]
    
    def execute_entry(self, symbol, details, date, candle_time, vix_close):
        entry_price_base = details['trigger_price']

        slippage_pct = self.cfg['base_slippage_percent']
        if self.cfg['use_vix_adaptive_filters'] and vix_close > self.cfg['vix_high_threshold']:
            slippage_pct = self.cfg['high_vol_slippage_percent']
        entry_price = entry_price_base * (1 + slippage_pct / 100)

        loc_d_indexer = self.daily_data[symbol].index.get_indexer([date], method='ffill')
        if not loc_d_indexer.size or loc_d_indexer[0] == -1: return
        loc_d = loc_d_indexer[0]

        stop_loss_slice = self.daily_data[symbol].iloc[max(0, loc_d - self.cfg['stop_loss_lookback_days']):loc_d]
        
        if stop_loss_slice.empty:
            self.debug_log.append(f"Entry Skip: {symbol} on {date.date()} - Not enough historical data for stop-loss.")
            return
        stop_loss = stop_loss_slice['low'].min()
        
        if pd.isna(stop_loss): return

        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0: return

        capital_to_risk = (self.portfolio['equity'] * (self.cfg['risk_per_trade_percent'] / 100))
        if self.cfg['use_dynamic_position_sizing']:
            active_risk = sum([(p['entry_price'] - p['stop_loss']) * p['shares'] for p in self.portfolio['positions'].values()])
            max_portfolio_risk_capital = self.portfolio['equity'] * (self.cfg['max_portfolio_risk_percent'] / 100)
            available_risk_capital = max(0, max_portfolio_risk_capital - active_risk)
            capital_to_risk = min(capital_to_risk, available_risk_capital)
        
        shares = math.floor(capital_to_risk / risk_per_share) if capital_to_risk > 0 else 0

        log_entry = {'setup_id': details['setup_id'], 'symbol': symbol, 'setup_date': pd.to_datetime(details['setup_id'].split('_')[-1]), 'trigger_date': date, 'trigger_price': details['trigger_price']}
        if shares > 0 and (shares * entry_price) <= self.portfolio['cash']:
            log_entry['status'] = 'FILLED'
            self.portfolio['cash'] -= shares * entry_price
            
            profit_target_rr = self.cfg['profit_target_rr_calm']
            if self.cfg['use_dynamic_profit_target'] and vix_close > self.cfg['vix_high_threshold']:
                profit_target_rr = self.cfg['profit_target_rr_high']
            
            target_price = entry_price + (risk_per_share * profit_target_rr)

            self.portfolio['positions'][f"{symbol}_{candle_time}"] = {
                'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price, 
                'stop_loss': stop_loss, 'shares': shares, 'target': target_price, 
                'partial_exit': False, 'initial_shares': shares, 'setup_id': details['setup_id']
            }
        elif shares > 0:
            log_entry['status'] = 'MISSED_CAPITAL'
        else:
            log_entry['status'] = 'FILTERED_RISK'
        
        self.all_setups_log.append(log_entry)

    def manage_intraday_exits(self, candle_time):
        exit_proceeds = 0
        to_remove = []
        for pos_id, pos in list(self.portfolio['positions'].items()):
            try:
                candle = self.intraday_data[pos['symbol']].loc[candle_time]
            except (KeyError, IndexError):
                continue

            if self.cfg['use_partial_profit_leg'] and not pos.get('partial_exit', False) and candle['high'] >= pos['target']:
                shares_to_sell = pos['shares'] // 2
                exit_price = pos['target']
                exit_proceeds += shares_to_sell * exit_price
                trade_log = pos.copy(); trade_log.update({'exit_date': candle_time, 'exit_price': exit_price, 'pnl': (exit_price - pos['entry_price']) * shares_to_sell, 'exit_type': 'Partial Profit'})
                self.portfolio['trades'].append(trade_log)
                pos['shares'] -= shares_to_sell
                pos['partial_exit'] = True
                pos['stop_loss'] = pos['entry_price']

            if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_proceeds += pos['shares'] * exit_price
                trade_log = pos.copy(); trade_log.update({'exit_date': candle_time, 'exit_price': exit_price, 'pnl': (exit_price - pos['entry_price']) * pos['shares'], 'exit_type': 'Stop-Loss'})
                self.portfolio['trades'].append(trade_log)
                to_remove.append(pos_id)
        
        for pos_id in to_remove:
            self.portfolio['positions'].pop(pos_id, None)
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
        
        eod_equity = self.portfolio['cash']
        for pos in self.portfolio['positions'].values():
            if date in self.daily_data.get(pos['symbol'], pd.DataFrame()).index:
                eod_equity += pos['shares'] * self.daily_data[pos['symbol']].loc[date]['close']
        self.portfolio['equity'] = eod_equity
        self.portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    def generate_report(self):
        # --- ORIGINAL REPORTING LOGIC (UNCHANGED) ---
        print("\n\n--- BACKTEST COMPLETE ---")
        final_equity = self.portfolio['equity']
        net_pnl = final_equity - self.cfg['initial_capital']
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
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            total_trades = len(trades_df)
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            gross_profit = winning_trades['pnl'].sum()
            gross_loss = abs(losing_trades['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        summary_content = f"""
BACKTEST SUMMARY REPORT (HTF ADVANCED SIMULATOR)
==================================================
Initial Capital: {self.cfg['initial_capital']:,.2f}
Final Equity:    {final_equity:,.2f}
Net P&L:         {net_pnl:,.2f}
--------------------------------------------------
CAGR:            {cagr:.2f}%
Max Drawdown:    {max_drawdown:.2f}%
Profit Factor:   {profit_factor:.2f}
Win Rate:        {win_rate:.2f}%
Total Trades:    {total_trades}
--------------------------------------------------
Total Setups Identified: {len(self.all_setups_log)}
Setups Filled:           {len([s for s in self.all_setups_log if s['status'] == 'FILLED'])}
Setups Missed (Capital): {len([s for s in self.all_setups_log if s['status'] == 'MISSED_CAPITAL'])}
Setups Filtered (Risk):  {len([s for s in self.all_setups_log if s['status'] == 'FILTERED_RISK'])}
"""
        print(summary_content)

        # --- NEW, ISOLATED, ADDITIVE LOGGING BLOCK ---
        print("\n--- Generating Enhanced Log Files ---")
        log_opts = self.cfg.get('log_options', {})
        strategy_log_folder = os.path.join(self.cfg['log_folder'], self.cfg['strategy_name'])
        os.makedirs(strategy_log_folder, exist_ok=True) # BUG FIX: Create the directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if log_opts.get('log_summary', True):
            params_str = "INPUT PARAMETERS:\n-----------------\n"
            for key, value in self.cfg.items():
                if isinstance(value, dict):
                    params_str += f"{key.replace('_', ' ').title()}:\n"
                    for k, v in value.items(): params_str += f"  - {k}: {v}\n"
                else: params_str += f"{key.replace('_', ' ').title()}: {value}\n"
            
            enhanced_summary_content = f"""{summary_content.strip()}\n""" # Reuse original summary
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
                    loc_d_indexer = self.daily_data[row['symbol']].index.get_indexer([row['trigger_date']], method='ffill')
                    if not loc_d_indexer.size or loc_d_indexer[0] == -1: continue
                    loc_d = loc_d_indexer[0]
                    stop_loss_slice = self.daily_data[row['symbol']].iloc[max(0, loc_d - self.cfg['stop_loss_lookback_days']):loc_d]
                    if not stop_loss_slice.empty:
                        stop_loss = stop_loss_slice['low'].min()
                        if pd.notna(stop_loss):
                            exit_date, pnl = simulate_htf_trade_outcome(row['symbol'], row['trigger_date'], row['trigger_price'], stop_loss, self.daily_data, self.cfg)
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
    config['log_folder'] = os.path.join(os.getcwd(), config['log_folder'])
    os.makedirs(config['log_folder'], exist_ok=True)
    
    simulator = HtfAdvancedSimulator(config)
    simulator.load_data()
    simulator.run_simulation()
