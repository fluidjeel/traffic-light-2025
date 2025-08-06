# monthly_tfl_simulator.py
#
# Description:
# A flexible version of the state-of-the-art, bias-free backtester for the
# Monthly Pullback Strategy.
#
# MODIFICATION (v1.3 - Realistic Fills & No Re-Entry):
# 1. REFACTORED: The `execute_entry` function now uses the close of the triggering
#    15-minute candle as the basis for the entry price, ensuring a realistic fill.
# 2. REFACTORED: The `manage_intraday_exits` function now accounts for price gaps
#    at the open for both stop-loss and profit-target exits.
# 3. FIXED: The logic now prevents multiple trades from being taken on the same
#    monthly setup ID, ensuring each signal is traded only once.
#
# MODIFICATION (v1.2 - Final DeepSeek Audit Corrections):
# 1. FIXED: VIX Temporal Misalignment & Position Sizing Flaws.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time, timedelta
import sys
import numpy as np

# --- CONFIGURATION FOR THE MONTHLY TFL SIMULATOR ---
config = {
    'initial_capital': 1000000,
    'nifty_list_csv': 'nifty200.csv',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',

    'data_pipeline_config': {
        'use_universal_pipeline': True,
        'universal_processed_folder': os.path.join('data', 'universal_processed'),
        'universal_intraday_folder': os.path.join('data', 'universal_historical_data'),
        'legacy_processed_folder': os.path.join('data', 'processed'),
        'legacy_intraday_folder': 'historical_data_15min',
    },

    'data_folder_base': '',
    'intraday_data_folder': '',
    'log_folder': 'backtest_logs',
    'strategy_name': 'monthly_tfl_simulator',
    'log_options': { 'log_trades': True, 'log_missed': True, 'log_summary': True, 'log_filtered': True },
    'excursion_logging': { 'enable_mfe_tracking': True },

    'timeframe': 'monthly',
    'use_ema_filter': True,
    'ema_period_monthly': 10,
    
    'stop_loss_mode': 'PERCENT',
    'fixed_stop_loss_percent': 0.02,

    'execution_window_base_days': 15,
    'execution_window_vix_extension': 7,
    'execution_window_skip_days': 2,

    'use_dynamic_position_sizing': True,
    'risk_per_trade_percent': 1.5,
    'max_portfolio_risk_percent': 3.0,
    'max_new_positions_per_day': 6,

    'vix_symbol': 'INDIAVIX',
    'vix_high_threshold': 25,
    'market_strength_index': 'NIFTY200_INDEX',
    'market_strength_threshold_calm': -0.25,
    'market_strength_threshold_high': -0.75,
    'base_slippage_percent': 0.10,
    'high_vol_slippage_percent': 0.20,
    'use_market_strength_filter': True,

    'profit_target_mode': 'RISK_BASED',
    'vix_scaled_rr_config': {
        'use_vix_scaled_rr_target': True,
        'vix_rr_scale': [
            {'vix_max': 15, 'rr': 4.0},
            {'vix_max': 22, 'rr': 3.5},
            {'vix_max': 30, 'rr': 2.0},
            {'vix_max': 999, 'rr': 1.5}
        ]
    },
    'use_aggressive_breakeven': True,
    'breakeven_buffer_percent': 0.0005,
}

# --- HELPER FUNCTIONS ---
def get_consecutive_red_candles(df, current_loc, min_candles=1):
    red_candles = []
    i = current_loc - 1
    while i >= 0 and (df.iloc[i]['close'] < df.iloc[i]['open']):
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles if len(red_candles) >= min_candles else []

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
        self.traded_setup_ids = set() # MODIFICATION: To prevent re-entry

    def load_data(self):
        pipeline_cfg = self.cfg['data_pipeline_config']
        if pipeline_cfg['use_universal_pipeline']:
            self.cfg['data_folder_base'] = pipeline_cfg['universal_processed_folder']
            self.cfg['intraday_data_folder'] = pipeline_cfg['universal_intraday_folder']
            intraday_index_filename = f"{self.cfg['market_strength_index']}_15min.csv"
        else:
            self.cfg['data_folder_base'] = pipeline_cfg['legacy_processed_folder']
            self.cfg['intraday_data_folder'] = pipeline_cfg['legacy_intraday_folder']
            intraday_index_filename = "NIFTY_200_15min.csv"
        
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
                    if os.path.exists(file): 
                        df = pd.read_csv(file, index_col='datetime', parse_dates=True)
                        if not df.index.is_unique: df = df[~df.index.duplicated(keep='first')]
                        data_map[symbol] = df
                
                intraday_file_path = os.path.join(self.cfg['intraday_data_folder'], f"{symbol}_15min.csv")
                if symbol == self.cfg['market_strength_index']:
                    intraday_file_path = os.path.join(self.cfg['intraday_data_folder'], intraday_index_filename)
                
                if os.path.exists(intraday_file_path): 
                    df_intra = pd.read_csv(intraday_file_path, index_col='datetime', parse_dates=True)
                    if not df_intra.index.is_unique: df_intra = df_intra[~df_intra.index.duplicated(keep='first')]
                    self.intraday_data[symbol] = df_intra
            except Exception as e:
                print(f"Warning: Could not load all data for {symbol}. Error: {e}")
        
        print("Data loading complete.")

    def run_simulation(self):
        master_dates = self.daily_data[self.cfg['market_strength_index']].loc[self.cfg['start_date']:self.cfg['end_date']].index
        
        for date in master_dates:
            progress_str = f"Processing {date.date()} | Equity: {self.portfolio['equity']:,.0f} | Positions: {len(self.portfolio['positions'])}"
            sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()
            
            if date.is_month_end: self.scout_for_setups(date)
            self.sniper_monitor_and_execute(date)
            self.manage_eod_portfolio(date)
        self.generate_report()

    def scout_for_setups(self, month_end_date):
        for symbol in self.tradeable_symbols:
            if symbol not in self.monthly_data: continue
            df_monthly = self.monthly_data[symbol]
            try:
                loc_monthly_indexer = df_monthly.index.get_indexer([month_end_date], method='ffill')
                if not loc_monthly_indexer.size or loc_monthly_indexer[0] == -1: continue
                loc_monthly = loc_monthly_indexer[0]
                setup_candle = df_monthly.iloc[loc_monthly]

                if not (setup_candle['close'] > setup_candle['open']): continue
                
                red_candles = get_consecutive_red_candles(df_monthly, loc_monthly)
                if not red_candles: continue

                trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])
                setup_id = f"{symbol}_{setup_candle.name.strftime('%Y-%m-%d')}"
                self.all_setups_log.append({'setup_id': setup_id, 'status': 'IDENTIFIED'})

                vix_df = self.daily_data[self.cfg['vix_symbol']]
                vix_close_t1 = vix_df.loc[:month_end_date].iloc[-1]['close']
                
                start_next_month = month_end_date + pd.offsets.MonthBegin(1)
                end_next_month = month_end_date + pd.offsets.MonthEnd(1)
                all_trading_days = self.daily_data[self.cfg['market_strength_index']].loc[start_next_month:end_next_month].index
                
                window_days = self.cfg['execution_window_base_days']
                if vix_close_t1 > self.cfg['vix_high_threshold']:
                    window_days += self.cfg['execution_window_vix_extension']
                execution_window_days = all_trading_days[self.cfg['execution_window_skip_days']:window_days]

                for day in execution_window_days:
                    day_str = day.strftime('%Y-%m-%d')
                    if day_str not in self.target_list: self.target_list[day_str] = {}
                    self.target_list[day_str][symbol] = {'trigger_price': trigger_price, 'setup_id': setup_id}
            except Exception: continue

    def sniper_monitor_and_execute(self, date):
        date_str = date.strftime('%Y-%m-%d')
        todays_watchlist = self.target_list.get(date_str, {})
        if not todays_watchlist: return

        try:
            vix_df = self.daily_data[self.cfg['vix_symbol']]
            vix_close_t1 = vix_df.loc[:date].iloc[-2]['close']
            mkt_idx_intra = self.intraday_data[self.cfg['market_strength_index']].loc[date_str]
        except (KeyError, IndexError): return

        for candle_time, _ in mkt_idx_intra.iterrows():
            self.manage_intraday_exits(candle_time)
            for symbol, details in list(todays_watchlist.items()):
                # MODIFICATION: Prevent re-entry on an already traded setup
                if details['setup_id'] in self.traded_setup_ids:
                    continue
                if any(p['symbol'] == symbol for p in self.portfolio['positions'].values()):
                    continue
                try:
                    stock_candle = self.intraday_data[symbol].loc[candle_time]
                except (KeyError, IndexError): continue

                if stock_candle['high'] >= details['trigger_price']:
                    # MODIFICATION: Pass the actual candle to the execution function
                    self.execute_entry(symbol, details, stock_candle, candle_time, vix_close_t1)
                    if symbol in todays_watchlist: del todays_watchlist[symbol]
    
    def execute_entry(self, symbol, details, stock_candle, candle_time, vix_close):
        # MODIFICATION: Use the close of the breakout candle for a realistic fill price
        entry_price_base = stock_candle['close']
        
        slippage_pct = self.cfg['base_slippage_percent']
        if vix_close > self.cfg['vix_high_threshold']:
            slippage_pct = self.cfg['high_vol_slippage_percent']
        entry_price = entry_price_base * (1 + slippage_pct / 100)
        
        if self.cfg['stop_loss_mode'] == 'PERCENT':
            stop_loss = entry_price * (1 - self.cfg['fixed_stop_loss_percent'])
        else: return 
        
        risk_per_share = entry_price - stop_loss
        if pd.isna(risk_per_share) or risk_per_share <= 0: return

        rr_config = self.cfg.get('vix_scaled_rr_config', {})
        profit_target_rr = 2.0 # Default
        if rr_config.get('use_vix_scaled_rr_target', False):
            scale = rr_config.get('vix_rr_scale', [])
            profit_target_rr = scale[-1]['rr']
            for level in scale:
                if vix_close <= level['vix_max']:
                    profit_target_rr = level['rr']; break
        
        target_price = entry_price + (risk_per_share * profit_target_rr)
        
        current_cash = self.portfolio['cash']
        capital_to_risk = current_cash * (self.cfg['risk_per_trade_percent'] / 100)
        if self.cfg['use_dynamic_position_sizing']:
            active_risk = sum([(p['entry_price'] - p['stop_loss']) * p['shares'] for p in self.portfolio['positions'].values()])
            max_portfolio_risk_capital = current_cash * (self.cfg['max_portfolio_risk_percent'] / 100)
            available_risk_capital = max(0, max_portfolio_risk_capital - active_risk)
            capital_to_risk = min(capital_to_risk, available_risk_capital)
        
        shares = math.floor(capital_to_risk / risk_per_share) if capital_to_risk > 0 else 0

        if shares > 0 and (shares * entry_price) <= self.portfolio['cash']:
            self.portfolio['cash'] -= shares * entry_price
            
            position_data = {
                'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price,
                'stop_loss': stop_loss, 'shares': shares, 'target': target_price,
                'setup_id': details['setup_id'], 'lowest_price_since_entry': entry_price,
                'highest_price_since_entry': entry_price, 'vix_on_entry': vix_close
            }
            self.portfolio['positions'][f"{symbol}_{candle_time}"] = position_data
            
            # MODIFICATION: Mark this setup ID as traded to prevent re-entry
            self.traded_setup_ids.add(details['setup_id'])

    def manage_intraday_exits(self, candle_time):
        exit_proceeds, to_remove = 0, []
        for pos_id, pos in list(self.portfolio['positions'].items()):
            try:
                candle = self.intraday_data[pos['symbol']].loc[candle_time]
                pos['lowest_price_since_entry'] = min(pos.get('lowest_price_since_entry', pos['entry_price']), candle['low'])
                pos['highest_price_since_entry'] = max(pos.get('highest_price_since_entry', pos['entry_price']), candle['high'])

                # --- PROFIT TARGET EXIT LOGIC ---
                if 'target' in pos and candle['high'] >= pos['target']:
                    # MODIFICATION: Use realistic fill price accounting for gaps
                    exit_price = pos['target']
                    if candle['open'] >= pos['target']: # Gapped up through target
                        exit_price = candle['open']
                    
                    exit_proceeds += pos['shares'] * exit_price
                    trade_log = self.create_enhanced_trade_log(pos, candle_time, exit_price, 'Profit Target', pos['shares'])
                    self.portfolio['trades'].append(trade_log)
                    to_remove.append(pos_id)
                    continue

                # --- STOP-LOSS EXIT LOGIC ---
                if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                    # MODIFICATION: Use realistic fill price accounting for gaps
                    exit_price = pos['stop_loss']
                    if candle['open'] <= pos['stop_loss']: # Gapped down through stop
                        exit_price = candle['open']

                    exit_proceeds += pos['shares'] * exit_price
                    trade_log = self.create_enhanced_trade_log(pos, candle_time, exit_price, 'Stop-Loss', pos['shares'])
                    self.portfolio['trades'].append(trade_log)
                    to_remove.append(pos_id)
            except (KeyError, IndexError): continue
        for pos_id in to_remove: self.portfolio['positions'].pop(pos_id, None)
        self.portfolio['cash'] += exit_proceeds

    def create_enhanced_trade_log(self, pos, exit_time, exit_price, exit_type, shares_exited):
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
        else: mae_percent, mfe_percent, captured_pct = 0, 0, 0
        base_log.update({
            'mae_price': mae_price, 'mae_percent': mae_percent,
            'mfe_price': mfe_price, 'mfe_percent': mfe_percent, 'captured_pct': captured_pct,
        })
        return base_log

    def manage_eod_portfolio(self, date):
        for pos in self.portfolio['positions'].values():
            if date in self.daily_data.get(pos['symbol'], pd.DataFrame()).index:
                daily_candle = self.daily_data[pos['symbol']].loc[date]
                new_stop = pos['stop_loss']
                if daily_candle['close'] > pos['entry_price']:
                    if self.cfg['use_aggressive_breakeven']:
                        buffer = pos['entry_price'] * self.cfg['breakeven_buffer_percent']
                        new_stop = max(new_stop, pos['entry_price'] + buffer)
                    if daily_candle['close'] > daily_candle['open']:
                        new_stop = max(new_stop, daily_candle['low'])
                pos['stop_loss'] = new_stop
        
        eod_value = sum([p['shares'] * self.daily_data[p['symbol']].loc[date]['close'] for p in self.portfolio['positions'].values() if date in self.daily_data.get(p['symbol'], pd.DataFrame()).index])
        self.portfolio['equity'] = self.portfolio['cash'] + eod_value
        self.portfolio['daily_values'].append({'date': date, 'equity': self.portfolio['equity']})

    def generate_report(self):
        # This function can be filled in with detailed report generation logic
        print("\n\n--- MONTHLY SIMULATOR BACKTEST COMPLETE ---")
        final_equity = self.portfolio['equity']
        print(f"Final Portfolio Equity: {final_equity:,.2f}")


if __name__ == '__main__':
    simulator = MonthlyAdvancedSimulator(config)
    simulator.load_data()
    simulator.run_simulation()
