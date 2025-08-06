# weekly_single_stock_analyzer.py
#
# Description:
# A specialized backtesting tool designed to run the Weekly HTF strategy
# against each stock in a universe individually. It compares the strategy's
# performance for a single stock against a simple Buy & Hold strategy for that same stock.
#
# MODIFICATION (v1.0 - Single-Stock Iteration):
# 1. NEW SCRIPT: Created a new script based on the weekly_tfl_simulator.py.
# 2. NEW LOGIC: The main execution loop now iterates through each symbol in the
#    Nifty 200 list, running a full backtest for each one.
# 3. NEW REPORTING: A consolidated summary report is generated at the end,
#    detailing the performance of the strategy vs. buy & hold for every stock.

import pandas as pd
import os
import math
from datetime import datetime, time as dt_time, timedelta
import sys
import numpy as np

# --- CONFIGURATION FOR THE WEEKLY SINGLE-STOCK ANALYZER ---
# This configuration is designed for a single-stock run, so some parameters
# like 'max_new_positions_per_day' are less critical.
config = {
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

    # --- Data Locations (These will be updated dynamically) ---
    'data_folder_base': '',
    'intraday_data_folder': '',
    'log_folder': 'backtest_logs',
    'strategy_name': 'weekly_tfl_simulator_single_stock',

    # --- Enhanced Logging Options ---
    'log_options': { 'log_trades': True, 'log_missed': True, 'log_summary': True, 'log_filtered': True },
    'excursion_logging': { 'enable_mfe_tracking': True },

    # --- Core HTF Strategy Parameters ---
    'timeframe': 'weekly',
    'use_ema_filter': False,
    'ema_period_htf': 30,

    # --- Portfolio & Risk Management ---
    'use_dynamic_position_sizing': True,
    'risk_per_trade_percent': 4.0,
    'max_portfolio_risk_percent': 15.0,
    'max_new_positions_per_day': 5,

    # --- Conviction Engine (Sniper Filters) ---
    'use_volume_projection': True,
    'volume_ma_period_daily': 20,
    'volume_multiplier_daily': 1.3,
    'volume_projection_thresholds': { dt_time(10, 0): 0.20, dt_time(11, 30): 0.45, dt_time(13, 0): 0.65, dt_time(14, 0): 0.85 },
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

    # --- Trade Management & Profit Taking (Trailing Logic) ---
    'use_partial_profit_leg': True,
    'use_dynamic_profit_target': True,
    'profit_target_rr_calm': 1.5,
    'profit_target_rr_high': 1.0,
    'use_aggressive_breakeven': True,
    'breakeven_buffer_percent': 0.0005,

    # -- Stop-Loss Configuration --
    'stop_loss_mode': 'LOOKBACK',
    'fixed_stop_loss_percent': 0.09,
    'stop_loss_lookback_days': 5,
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
    if applicable_threshold == 0: return False, 0, 0
    required_volume = target_daily_volume * applicable_threshold
    return cumulative_volume >= required_volume, cumulative_volume, required_volume

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
        
    # Assuming 252 trading days in a year
    cagr = ((final_price / initial_price) ** (1 / (len(df) / 252)) - 1) * 100
    
    return ((final_price - initial_price) / initial_price) * 100, cagr

# --- MAIN SIMULATOR CLASS ---
class WeeklySingleStockSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
        self.daily_data, self.intraday_data, self.htf_data = {}, {}, {}
        self.all_setups_log, self.target_list, self.debug_log, self.filtered_log = [], {}, [], []
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
                for tf, data_map in [('daily', self.daily_data), (self.cfg['timeframe'], self.htf_data)]:
                    file = os.path.join(self.cfg['data_folder_base'], tf, f"{symbol}_{tf}_with_indicators.csv")
                    if os.path.exists(file): data_map[symbol] = pd.read_csv(file, index_col='datetime', parse_dates=True)
                
                intraday_filename_to_load = f"{symbol}_15min.csv"
                if symbol == self.cfg['market_strength_index']:
                    intraday_filename_to_load = intraday_index_filename
                
                intraday_file_path = os.path.join(self.cfg['intraday_data_folder'], intraday_filename_to_load)
                if os.path.exists(intraday_file_path): 
                    self.intraday_data[symbol] = pd.read_csv(intraday_file_path, index_col='datetime', parse_dates=True)
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
            if date.weekday() == 4: self.scout_for_setups(date)
            self.sniper_monitor_and_execute(date)
            self.manage_eod_portfolio(date)
        
        return self.generate_report()

    def scout_for_setups(self, friday_date):
        symbol = self.symbol_to_backtest
        if symbol not in self.htf_data or symbol not in self.daily_data: return
        df_htf, df_daily = self.htf_data[symbol], self.daily_data[symbol]
        if df_htf.empty: return
        try:
            loc_htf_indexer = df_htf.index.get_indexer([friday_date], method='ffill')
            if not loc_htf_indexer.size or loc_htf_indexer[0] == -1: return
            loc_htf = loc_htf_indexer[0]
            setup_candle = df_htf.iloc[loc_htf]

            is_green = setup_candle['close'] > setup_candle['open']
            above_ema = True
            if self.cfg.get('use_ema_filter', True):
                above_ema = setup_candle['close'] > setup_candle.get(f"ema_{self.cfg['ema_period_htf']}", np.inf)
            if not (is_green and above_ema): return

            red_candles = get_consecutive_red_candles(df_htf, loc_htf)
            if not red_candles: return

            trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])
            if friday_date not in df_daily.index: return
            
            vol_sma = df_daily.loc[friday_date, f"volume_{self.cfg['volume_ma_period_daily']}_sma"]
            target_daily_volume = vol_sma * self.cfg['volume_multiplier_daily']
            if pd.isna(target_daily_volume): return

            setup_details = {'trigger_price': trigger_price, 'target_daily_volume': target_daily_volume, 'setup_id': f"{symbol}_{setup_candle.name.strftime('%Y-%m-%d')}"}
            
            for i in range(1, 6):
                next_day = friday_date + timedelta(days=i)
                next_day_str = next_day.strftime('%Y-%m-%d')
                if next_day_str not in self.target_list: self.target_list[next_day_str] = {}
                self.target_list[next_day_str][symbol] = setup_details
        except Exception as e:
            self.debug_log.append(f"ERROR in scout_for_setups for {symbol} on {friday_date.date()}: {e}")

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
            if mkt_idx_intra.empty: return
            mkt_open = mkt_idx_intra.iloc[0]['open']
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
                    stock_intra_df = self.intraday_data[symbol].loc[date_str]
                    stock_candle = stock_intra_df.loc[candle_time]
                    stock_open = stock_intra_df.iloc[0]['open']
                except (KeyError, IndexError): continue

                if stock_candle['high'] >= details['trigger_price']:
                    log_template = {'symbol': symbol, 'timestamp': candle_time, 'trigger_price': details['trigger_price'], 'setup_id': details['setup_id']}
                    if self.cfg['use_volume_projection']:
                        cum_vol = stock_intra_df.loc[:candle_time, 'volume'].sum()
                        vol_ok, actual_vol, req_vol = check_volume_projection(candle_time, cum_vol, details['target_daily_volume'], self.cfg['volume_projection_thresholds'])
                        if not vol_ok: self.filtered_log.append({**log_template, 'filter_type': 'Volume Projection', 'actual': f"{actual_vol:,.0f}", 'expected': f"{req_vol:,.0f}"}); continue
                    
                    if self.cfg['use_vix_adaptive_filters']:
                        if candle_idx > 0:
                            prev_mkt_candle = mkt_idx_intra.iloc[candle_idx - 1]
                            mkt_strength = (prev_mkt_candle['close'] / mkt_open - 1) * 100
                            threshold = self.cfg['market_strength_threshold_high'] if vix_close_t1 > self.cfg['vix_high_threshold'] else self.cfg['market_strength_threshold_calm']
                            if mkt_strength < threshold: self.filtered_log.append({**log_template, 'filter_type': 'Market Strength', 'actual': f"{mkt_strength:.2f}%", 'expected': f">{threshold:.2f}%"}); continue

                    if self.cfg['use_intraday_rs_filter']:
                        if candle_idx > 0:
                            prev_mkt_candle = mkt_idx_intra.iloc[candle_idx - 1]
                            prev_stock_candle = stock_intra_df.iloc[candle_idx - 1]
                            mkt_strength = (prev_mkt_candle['close'] / mkt_open - 1) * 100
                            stock_rs = (prev_stock_candle['close'] / stock_open - 1) * 100
                            if stock_rs < mkt_strength: self.filtered_log.append({**log_template, 'filter_type': 'Intraday RS', 'actual': f"{stock_rs:.2f}%", 'expected': f">{mkt_strength:.2f}%"}); continue
                    
                    self.execute_entry(symbol, details, date, candle_time, vix_close_t1)
                    todays_new_positions += 1
                    if symbol in todays_watchlist: del todays_watchlist[symbol]
    
    def execute_entry(self, symbol, details, date, candle_time, vix_close):
        entry_price_base = details['trigger_price']
        slippage_pct = self.cfg['base_slippage_percent']
        if self.cfg['use_vix_adaptive_filters'] and vix_close > self.cfg['vix_high_threshold']:
            slippage_pct = self.cfg['high_vol_slippage_percent']
        entry_price = entry_price_base * (1 + slippage_pct / 100)

        if self.cfg.get('stop_loss_mode') == 'PERCENT':
            stop_loss = entry_price * (1 - self.cfg['fixed_stop_loss_percent'])
        else: # Default to 'LOOKBACK'
            loc_d_indexer = self.daily_data[symbol].index.get_indexer([date], method='ffill')
            if not loc_d_indexer.size or loc_d_indexer[0] == -1: return
            loc_d = loc_d_indexer[0]
            stop_loss_slice = self.daily_data[symbol].iloc[max(0, loc_d - self.cfg['stop_loss_lookback_days']):loc_d]
            if stop_loss_slice.empty: return
            stop_loss = stop_loss_slice['low'].min()
        
        if pd.isna(stop_loss): return
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0: return

        profit_target_rr = self.cfg['profit_target_rr_calm']
        if self.cfg['use_dynamic_profit_target'] and vix_close > self.cfg['vix_high_threshold']:
            profit_target_rr = self.cfg['profit_target_rr_high']
        target_price = entry_price + (risk_per_share * profit_target_rr)

        # --- CRITICAL FIX: Use real-time cash for all risk calculations ---
        current_cash = self.portfolio['cash']
        capital_to_risk = current_cash * (self.cfg['risk_per_trade_percent'] / 100)
        if self.cfg['use_dynamic_position_sizing']:
            active_risk = sum([(p['entry_price'] - p['stop_loss']) * p['shares'] for p in self.portfolio['positions'].values()])
            max_portfolio_risk_capital = current_cash * (self.cfg['max_portfolio_risk_percent'] / 100)
            available_risk_capital = max(0, max_portfolio_risk_capital - active_risk)
            capital_to_risk = min(capital_to_risk, available_risk_capital)
        # --- END CRITICAL FIX ---
        
        shares = math.floor(capital_to_risk / risk_per_share) if capital_to_risk > 0 else 0

        log_entry = {'setup_id': details['setup_id'], 'symbol': symbol, 'setup_date': pd.to_datetime(details['setup_id'].split('_')[-1]), 'trigger_date': date, 'trigger_price': details['trigger_price']}
        if shares > 0 and (shares * entry_price) <= self.portfolio['cash']:
            log_entry['status'] = 'FILLED'
            self.portfolio['cash'] -= shares * entry_price
            self.portfolio['positions'][f"{symbol}_{candle_time}"] = {'symbol': symbol, 'entry_date': candle_time, 'entry_price': entry_price, 'stop_loss': stop_loss, 'shares': shares, 'target': target_price, 'partial_exit': False, 'initial_shares': shares, 'setup_id': details['setup_id'], 'lowest_price_since_entry': entry_price}
        elif shares > 0:
            log_entry['status'] = 'MISSED_CAPITAL'
        else:
            log_entry['status'] = 'FILTERED_RISK'
        self.all_setups_log.append(log_entry)

    def manage_intraday_exits(self, candle_time):
        exit_proceeds, to_remove = 0, []
        for pos_id, pos in list(self.portfolio['positions'].items()):
            try:
                candle = self.intraday_data[pos['symbol']].loc[candle_time]
                pos['lowest_price_since_entry'] = min(pos.get('lowest_price_since_entry', pos['entry_price']), candle['low'])
            except (KeyError, IndexError):
                continue

            if self.cfg['use_partial_profit_leg'] and not pos.get('partial_exit', False) and candle['high'] >= pos['target']:
                shares_to_sell = pos['shares'] // 2
                exit_price = pos['target']
                exit_proceeds += shares_to_sell * exit_price
                mae_price = pos['lowest_price_since_entry']
                mae_percent = ((pos['entry_price'] - mae_price) / pos['entry_price']) * 100
                trade_log = pos.copy(); trade_log.update({'exit_date': candle_time, 'exit_price': exit_price, 'pnl': (exit_price - pos['entry_price']) * shares_to_sell, 'exit_type': 'Partial Profit', 'mae_price': mae_price, 'mae_percent': mae_percent})
                self.portfolio['trades'].append(trade_log)
                pos['shares'] -= shares_to_sell
                pos['partial_exit'] = True
                pos['stop_loss'] = pos['entry_price']

            if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_proceeds += pos['shares'] * exit_price
                mae_price = pos['lowest_price_since_entry']
                mae_percent = ((pos['entry_price'] - mae_price) / pos['entry_price']) * 100
                trade_log = pos.copy(); trade_log.update({'exit_date': candle_time, 'exit_price': exit_price, 'pnl': (exit_price - pos['entry_price']) * pos['shares'], 'exit_type': 'Stop-Loss', 'mae_price': mae_price, 'mae_percent': mae_percent})
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
                        buffer = pos['entry_price'] * self.cfg['breakeven_buffer_percent']
                        new_stop = max(new_stop, pos['entry_price'] + buffer)
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
        simulator = WeeklySingleStockSimulator(config)
        
        if simulator.load_data():
            result = simulator.run_simulation()
            all_results.append(result)

    final_results_df = pd.DataFrame(all_results)
    
    # Calculate Buy & Hold returns for comparison
    master_simulator = WeeklySingleStockSimulator(config)
    if master_simulator.load_data():
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
    report_filename = os.path.join(log_folder, f"{timestamp}_weekly_single_stock_report.csv")
    final_results_df.to_csv(report_filename, index=False)
    
    print(f"\n\n--- CONSOLIDATED REPORT SAVED TO '{report_filename}' ---")
    print(final_results_df.to_string())
