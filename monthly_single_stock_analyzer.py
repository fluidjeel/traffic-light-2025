# monthly_single_stock_analyzer.py
#
# Description:
# A specialized backtesting tool designed to run the Monthly Pullback Strategy
# against each stock in a universe individually. It compares the strategy's
# performance for a single stock against a simple Buy & Hold strategy for that same stock.
#
# MODIFICATION (v1.0 - Single-Stock Iteration):
# 1. NEW SCRIPT: Created a new script based on the monthly_tfl_simulator.py.
# 2. NEW LOGIC: The main execution loop now iterates through each symbol in the
#    Nifty 200 list, running a full backtest for each one.
# 3. NEW REPORTING: A consolidated summary report is generated at the end,
#    detailing the performance of the strategy vs. buy & hold for every stock.
# 4. FIXED: Removed dynamic trailing stop parameters and logic as requested.
# 5. FIXED: Corrected a syntax error in the volume filter calculation.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time, timedelta
import sys
import numpy as np

# --- CONFIGURATION FOR THE MONTHLY SINGLE-STOCK ANALYZER ---
# This configuration is designed for a single-stock run, so some parameters
# like 'max_new_positions_per_day' are less critical.
config = {
    # --- General Backtest Parameters ---
    'initial_capital': 1000000,
    'nifty_list_csv': 'nifty200.csv',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'symbol_to_backtest': None, # This will be set dynamically

    # --- DATA PIPELINE CONFIGURATION ---
    'data_pipeline_config': {
        'use_universal_pipeline': True, # MASTER TOGGLE: Set to False to use legacy paths

        # --- Universal Data Paths ---
        'universal_processed_folder': os.path.join('data', 'universal_processed'),
        'universal_intraday_folder': os.path.join('data', 'universal_historical_data'),
        
        # --- Legacy Data Paths ---
        'legacy_processed_folder': os.path.join('data', 'processed'),
        'legacy_intraday_folder': 'historical_data_15min',
    },

    # --- Data & Logging Paths (These will be updated dynamically) ---
    'data_folder_base': '',
    'intraday_data_folder': '',
    'log_folder': 'backtest_logs',
    'strategy_name': 'monthly_tfl_simulator_single_stock',

    # --- Enhanced Logging Options ---
    'log_options': { 'log_trades': True, 'log_missed': True, 'log_summary': True, 'log_filtered': True },
    'excursion_logging': { 'enable_mfe_tracking': True },

    # --- Core Monthly Strategy Parameters (Scout) ---
    'timeframe': 'monthly',
    'use_ema_filter': False,
    'ema_period_monthly': 10,
    'use_monthly_volume_filter': False,
    'volume_ma_period_monthly': 12,
    'volume_multiplier_monthly': 1.2,
    
    # --- Stop-Loss Configuration ---
    'stop_loss_mode': 'PERCENT',
    'fixed_stop_loss_percent': 0.04,
    
    # --- Adaptive Execution Window (Sniper) ---
    'execution_window_base_days': 15,
    'execution_window_vix_extension': 7,
    'execution_window_skip_days': 2,

    # --- Portfolio & Risk Management ---
    'use_dynamic_position_sizing': True,
    'risk_per_trade_percent': 4.0,
    'max_portfolio_risk_percent': 6.0,
    'max_new_positions_per_day': 6,

    # --- Conviction Engine (Sniper Filters) ---
    'vix_symbol': 'INDIAVIX',
    'vix_high_threshold': 25,
    'market_strength_index': 'NIFTY200_INDEX',
    'market_strength_threshold_calm': -0.25,
    'market_strength_threshold_high': -0.75,
    'base_slippage_percent': 0.10,
    'high_vol_slippage_percent': 0.20,
    'use_market_strength_filter': False,

    # --- Trade Management & Profit Taking ---
    'profit_target_mode': 'RISK_BASED',
    'use_partial_profit_leg': False,
    
    'vix_scaled_rr_config': {
        'use_vix_scaled_rr_target': True,
        'vix_rr_scale': [
            {'vix_max': 15, 'rr': 3.0},
            {'vix_max': 22, 'rr': 2.5},
            {'vix_max': 30, 'rr': 2.0},
            {'vix_max': 999, 'rr': 1.5}
        ]
    },
    'profit_target_rr_calm': 2.0,
    'profit_target_rr_high': 1.5,
    
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

def calculate_buy_and_hold_return(symbol, daily_data, start_date, end_date):
    if symbol not in daily_data:
        return 0, 0
    
    df = daily_data[symbol].loc[start_date:end_date]
    if df.empty:
        return 0, 0
    
    initial_price = df.iloc[0]['open']
    final_price = df.iloc[-1]['close']
    
    if initial_price == 0:
        return 0, 0
        
    cagr = ((final_price / initial_price) ** (1 / (len(df) / 252)) - 1) * 100
    
    return ((final_price - initial_price) / initial_price) * 100, cagr


# --- MAIN SIMULATOR CLASS ---
# This class is a simplified version for single-stock backtesting.
class MonthlySingleStockSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
        self.daily_data, self.intraday_data, self.monthly_data = {}, {}, {}
        self.target_list = {}
        self.all_setups_log = []
        self.filtered_log = []
        self.symbol_to_backtest = cfg['symbol_to_backtest']

    def load_data(self):
        pipeline_cfg = self.cfg['data_pipeline_config']
        if pipeline_cfg['use_universal_pipeline']:
            print(f"--- Loading UNIVERSAL Data for {self.symbol_to_backtest} ---")
            self.cfg['data_folder_base'] = pipeline_cfg['universal_processed_folder']
            self.cfg['intraday_data_folder'] = pipeline_cfg['universal_intraday_folder']
            intraday_index_filename = "{symbol}_15min.csv".format(symbol=self.cfg['market_strength_index'])
        else:
            print(f"--- Loading LEGACY Data for {self.symbol_to_backtest} ---")
            self.cfg['data_folder_base'] = pipeline_cfg['legacy_processed_folder']
            self.cfg['intraday_data_folder'] = pipeline_cfg['legacy_intraday_folder']
            intraday_index_filename = "NIFTY_200_15min.csv"
        
        symbols_to_load = [self.symbol_to_backtest, self.cfg['market_strength_index'], self.cfg['vix_symbol']]
        for symbol in symbols_to_load:
            try:
                for tf, data_map in [('daily', self.daily_data), ('monthly', self.monthly_data)]:
                    file = os.path.join(self.cfg['data_folder_base'], tf, f"{symbol}_{tf}_with_indicators.csv")
                    if os.path.exists(file): 
                        df = pd.read_csv(file, index_col='datetime', parse_dates=True)
                        if not df.index.is_unique:
                            df = df[~df.index.duplicated(keep='first')]
                        data_map[symbol] = df
                
                intraday_filename_to_load = f"{symbol}_15min.csv"
                if symbol == self.cfg['market_strength_index']:
                    intraday_filename_to_load = intraday_index_filename
                
                intraday_file_path = os.path.join(self.cfg['intraday_data_folder'], intraday_filename_to_load)
                if os.path.exists(intraday_file_path): 
                    df_intra = pd.read_csv(intraday_file_path, index_col='datetime', parse_dates=True)
                    if not df_intra.index.is_unique:
                        df_intra = df_intra[~df_intra.index.duplicated(keep='first')]
                    self.intraday_data[symbol] = df_intra
            except Exception as e:
                print(f"Warning: Could not load all data for {symbol}. Error: {e}")

        if self.symbol_to_backtest not in self.daily_data:
            print(f"FATAL ERROR: Could not load data for {self.symbol_to_backtest}. Skipping.")
            return False
        
        return True

    def run_simulation(self):
        if not self.daily_data or self.cfg['market_strength_index'] not in self.daily_data:
            print("Error: Market index data not loaded. Skipping simulation."); return
        
        master_dates = self.daily_data[self.cfg['market_strength_index']].loc[self.cfg['start_date']:self.cfg['end_date']].index
        
        for date in master_dates:
            if date.is_month_end: self.scout_for_setups(date)
            self.sniper_monitor_and_execute(date)
            self.manage_eod_portfolio(date)
        
        return self.generate_report()

    def scout_for_setups(self, month_end_date):
        symbol = self.symbol_to_backtest
        if symbol not in self.monthly_data or symbol not in self.daily_data: return
        df_monthly = self.monthly_data[symbol]
        if df_monthly.empty: return
        try:
            loc_monthly_indexer = df_monthly.index.get_indexer([month_end_date], method='ffill')
            if not loc_monthly_indexer.size or loc_monthly_indexer[0] == -1: return
            loc_monthly = loc_monthly_indexer[0]
            setup_candle = df_monthly.iloc[loc_monthly]

            is_green = setup_candle['close'].item() > setup_candle['open'].item()
            above_ema = True
            if self.cfg.get('use_ema_filter', False):
                above_ema = setup_candle['close'].item() > setup_candle.get(f"ema_{self.cfg['ema_period_monthly']}", np.inf).item()
            
            volume_ok = True
            if self.cfg.get('use_monthly_volume_filter', False):
                volume_ok = setup_candle['volume'].item() > (setup_candle[f"volume_{self.cfg['volume_ma_period_monthly']}_sma"].item() * self.cfg['volume_multiplier_monthly'])

            if not (is_green and above_ema and volume_ok): return
            
            red_candles = get_consecutive_red_candles(df_monthly, loc_monthly, min_candles=1)
            if not red_candles: return

            trigger_price = max([c['high'].item() for c in red_candles] + [setup_candle['high'].item()])
            monthly_atr = 0
            if self.cfg['stop_loss_mode'] == 'ATR_BASED':
                monthly_atr = setup_candle[f"atr_{self.cfg['atr_period_monthly']}"].item()
            if pd.isna(monthly_atr): return

            setup_id = f"{symbol}_{setup_candle.name.strftime('%Y-%m-%d')}"
            self.all_setups_log.append({'setup_id': setup_id, 'status': 'IDENTIFIED'})

            vix_df = self.daily_data[self.cfg['vix_symbol']]
            vix_loc = vix_df.index.get_loc(month_end_date)
            if vix_loc < 1: return
            vix_close_t1 = vix_df.iloc[vix_loc - 1]['close'].item()
            
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
                self.target_list[day_str][symbol] = {'trigger_price': trigger_price, 'monthly_atr': monthly_atr, 'setup_id': setup_id}
        except Exception: return

    def sniper_monitor_and_execute(self, date):
        date_str = date.strftime('%Y-%m-%d')
        todays_watchlist = self.target_list.get(date_str, {})
        if not todays_watchlist: return

        try:
            vix_df = self.daily_data[self.cfg['vix_symbol']]
            vix_loc = vix_df.index.get_loc(date)
            if vix_loc < 1: return
            vix_close_t1 = vix_df.iloc[vix_loc - 1]['close'].item()
            
            mkt_idx_intra = self.intraday_data[self.cfg['market_strength_index']].loc[date_str]
            mkt_open = mkt_idx_intra.iloc[0]['open'].item()

        except (KeyError, IndexError): return

        todays_new_positions = 0
        for candle_idx, (candle_time, mkt_candle) in enumerate(mkt_idx_intra.iterrows()):
            self.manage_intraday_exits(candle_time)
            if todays_new_positions >= self.cfg['max_new_positions_per_day']: continue
            for symbol, details in list(todays_watchlist.items()):
                if any(p['symbol'] == symbol for p in self.portfolio['positions'].values()):
                    if symbol in todays_watchlist: del todays_watchlist[symbol]
                    continue
                try:
                    stock_candle = self.intraday_data[symbol].loc[candle_time]
                except (KeyError, IndexError): continue

                if stock_candle['high'].item() >= details['trigger_price']:
                    log_template = {'symbol': symbol, 'timestamp': candle_time, 'trigger_price': details['trigger_price'], 'setup_id': details['setup_id']}
                    if self.cfg.get('use_market_strength_filter', False):
                        if candle_idx > 0:
                            prev_mkt_candle = mkt_idx_intra.iloc[candle_idx - 1]
                            mkt_strength = (prev_mkt_candle['close'].item() / mkt_open - 1) * 100
                            threshold = self.cfg['market_strength_threshold_high'] if vix_close_t1 > self.cfg['vix_high_threshold'] else self.cfg['market_strength_threshold_calm']
                            if mkt_strength < threshold:
                                self.filtered_log.append({**log_template, 'filter_type': 'Market Strength', 'actual': f"{mkt_strength:.2f}%", 'expected': f">{threshold:.2f}%"})
                                continue

                    self.execute_entry(symbol, details, candle_time, vix_close_t1)
                    todays_new_positions += 1
                    if symbol in todays_watchlist: del todays_watchlist[symbol]
    
    def execute_entry(self, symbol, details, candle_time, vix_close):
        entry_price_base = details['trigger_price']
        
        slippage_pct = self.cfg['base_slippage_percent']
        if vix_close > self.cfg['vix_high_threshold']:
            slippage_pct = self.cfg['high_vol_slippage_percent']
        entry_price = entry_price_base * (1 + slippage_pct / 100)
        
        if self.cfg['stop_loss_mode'] == 'PERCENT':
            stop_loss_actual = entry_price * (1 - self.cfg['fixed_stop_loss_percent'])
        else: return 
        
        risk_per_share_actual = entry_price - stop_loss_actual
        if pd.isna(risk_per_share_actual) or risk_per_share_actual <= 0: return

        rr_config = self.cfg.get('vix_scaled_rr_config', {})
        if rr_config.get('use_vix_scaled_rr_target', False):
            scale = rr_config.get('vix_rr_scale', [])
            profit_target_rr = scale[-1]['rr']
            for level in scale:
                if vix_close <= level['vix_max']:
                    profit_target_rr = level['rr']
                    break
        else:
            profit_target_rr = self.cfg['profit_target_rr_calm'] if vix_close <= self.cfg['vix_high_threshold'] else self.cfg['profit_target_rr_high']

        target_price = entry_price + (risk_per_share_actual * profit_target_rr)
        
        current_cash = self.portfolio['cash']
        capital_to_risk = current_cash * (self.cfg['risk_per_trade_percent'] / 100)
        if self.cfg['use_dynamic_position_sizing']:
            active_risk = sum([(p['entry_price'] - p['stop_loss']) * p['shares'] for p in self.portfolio['positions'].values()])
            max_portfolio_risk_capital = current_cash * (self.cfg['max_portfolio_risk_percent'] / 100)
            available_risk_capital = max(0, max_portfolio_risk_capital - active_risk)
            capital_to_risk = min(capital_to_risk, available_risk_capital)
        
        shares = math.floor(capital_to_risk / risk_per_share_actual) if capital_to_risk > 0 else 0

        log_entry = {'setup_id': details['setup_id'], 'symbol': symbol, 'setup_date': pd.to_datetime(details['setup_id'].split('_')[-1]), 'trigger_date': candle_time.date(), 'trigger_price': details['trigger_price']}
        if shares > 0 and (shares * entry_price) <= self.portfolio['cash']:
            log_entry['status'] = 'FILLED'
            self.portfolio['cash'] -= shares * entry_price
            
            position_data = {
                'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price,
                'stop_loss': stop_loss_actual, 'shares': shares, 'target': target_price,
                'initial_shares': shares, 'setup_id': details['setup_id'],
                'lowest_price_since_entry': entry_price,
                'vix_close': vix_close,
            }
            if self.cfg['excursion_logging']['enable_mfe_tracking']:
                position_data['highest_price_since_entry'] = entry_price

            self.portfolio['positions'][f"{symbol}_{candle_time}"] = position_data
        elif shares > 0:
            log_entry['status'] = 'MISSED_CAPITAL'
        else:
            log_entry['status'] = 'FILTERED_RISK'
        
        existing_log = next((item for item in self.all_setups_log if item['setup_id'] == details['setup_id']), None)
        if existing_log:
            existing_log.update(log_entry)

    def manage_intraday_exits(self, candle_time):
        exit_proceeds, to_remove = 0, []
        for pos_id, pos in list(self.portfolio['positions'].items()):
            try:
                candle = self.intraday_data[pos['symbol']].loc[candle_time]
                
                pos['lowest_price_since_entry'] = min(pos.get('lowest_price_since_entry', pos['entry_price']), candle['low'].item())
                if self.cfg['excursion_logging']['enable_mfe_tracking']:
                    pos['highest_price_since_entry'] = max(pos.get('highest_price_since_entry', pos['entry_price']), candle['high'].item())

                if 'target' in pos and candle['high'].item() >= pos['target']:
                    if self.cfg.get('use_partial_profit_leg', False):
                        shares_to_sell = pos['shares'] // 2
                        exit_price = pos['target']
                        exit_proceeds += shares_to_sell * exit_price
                        trade_log = self.create_enhanced_trade_log(pos, candle_time, exit_price, 'Partial Profit', shares_to_sell)
                        self.portfolio['trades'].append(trade_log)
                        pos.update({'shares': pos['shares'] - shares_to_sell, 'partial_exit': True, 'stop_loss': pos['entry_price']})
                    else:
                        exit_price = pos['target']
                        exit_proceeds += pos['shares'] * exit_price
                        trade_log = self.create_enhanced_trade_log(pos, candle_time, exit_price, 'Profit Target', pos['shares'])
                        self.portfolio['trades'].append(trade_log)
                        to_remove.append(pos_id)
                        continue

                if pos['shares'] > 0 and candle['low'].item() <= pos['stop_loss']:
                    exit_price = pos['stop_loss']
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
        if self.cfg['excursion_logging']['enable_mfe_tracking']:
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
                
                if daily_candle['close'].item() > pos['entry_price']:
                    if self.cfg['use_aggressive_breakeven'] and not pos.get('partial_exit', False):
                        buffer = pos['entry_price'] * self.cfg['breakeven_buffer_percent']
                        new_stop = max(new_stop, pos['entry_price'] + buffer)
                    
                    if daily_candle['close'].item() > daily_candle['open'].item():
                        new_stop = max(new_stop, daily_candle['low'].item())
                        
                pos['stop_loss'] = new_stop
        
        eod_value_of_positions = sum([p['shares'] * self.daily_data[p['symbol']].loc[date]['close'].item() for p in self.portfolio['positions'].values() if date in self.daily_data.get(p['symbol'], pd.DataFrame()).index])
        self.portfolio['equity'] = self.portfolio['cash'] + eod_value_of_positions
        self.portfolio['daily_values'].append({'date': date, 'equity': self.portfolio['equity']})

    def generate_report(self):
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
        
        summary = {
            'Symbol': self.symbol_to_backtest,
            'Final Equity': final_equity,
            'CAGR': cagr,
            'Max Drawdown': max_drawdown,
            'Profit Factor': profit_factor,
            'Win Rate': win_rate,
            'Total Trades': total_trades,
            'Initial Capital': self.cfg['initial_capital']
        }
        return summary

# --- Main execution loop for all symbols ---
if __name__ == '__main__':
    nifty_list_path = config['nifty_list_csv']
    try:
        nifty_symbols = pd.read_csv(nifty_list_path)['Symbol'].tolist()
    except FileNotFoundError:
        print(f"FATAL ERROR: Symbols file not found at '{nifty_list_path}'. Exiting."); sys.exit()

    all_results = []
    
    print("Starting individual stock backtests...")
    for symbol in nifty_symbols:
        print(f"\n--- Running backtest for {symbol} ---")
        config['symbol_to_backtest'] = symbol
        simulator = MonthlySingleStockSimulator(config)
        
        if simulator.load_data():
            result = simulator.run_simulation()
            all_results.append(result)

    final_results_df = pd.DataFrame(all_results)
    
    # Calculate Buy & Hold returns for comparison
    master_simulator = MonthlySingleStockSimulator(config)
    master_simulator.load_data() # Loads index and VIX data for buy and hold calculation
    buy_and_hold_results = {}
    for symbol in nifty_symbols:
        bh_return, bh_cagr = calculate_buy_and_hold_return(
            symbol, 
            master_simulator.daily_data, 
            config['start_date'], 
            config['end_date']
        )
        buy_and_hold_results[symbol] = {'bh_return': bh_return, 'bh_cagr': bh_cagr}

    final_results_df['BH Return %'] = final_results_df['Symbol'].map(lambda x: buy_and_hold_results.get(x, {}).get('bh_return', 0))
    final_results_df['BH CAGR %'] = final_results_df['Symbol'].map(lambda x: buy_and_hold_results.get(x, {}).get('bh_cagr', 0))
    final_results_df['Strategy vs. BH (%)'] = final_results_df['CAGR'] - final_results_df['BH CAGR %']
    
    # Save the consolidated report
    log_folder = os.path.join('backtest_logs', 'single_stock_analysis')
    os.makedirs(log_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(log_folder, f"{timestamp}_single_stock_report.csv")
    final_results_df.to_csv(report_filename, index=False)
    
    print(f"\n\n--- CONSOLIDATED REPORT SAVED TO '{report_filename}' ---")
    print(final_results_df.to_string())
