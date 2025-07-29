# htf_simulator_advanced.py
#
# Description:
# This is the new, state-of-the-art, bias-free backtesting simulator for the 
# Nifty 200 HTF (Higher Timeframe) Pullback Strategy.
#
# Architecture:
# It uses a "Scout and Sniper" model to eliminate lookahead bias.
# 1. Scout (EOD Friday): Scans for valid weekly setups based on the definitive
#    strategy rules and generates a target list for the upcoming week.
# 2. Sniper (Intraday Mon-Fri): Monitors the target list and executes entries
#    only when a price breakout is validated by a sophisticated, real-time
#    conviction engine ported from the advanced daily simulator.
#
# Key Enhancements:
# - Correct Bias-Free Setup Logic: Implements the exact strategy rules from
#   project_implementation.md, removing benchmark code contradictions.
# - Advanced Conviction Engine: Incorporates Time-Anchored Volume Projection,
#   VIX-Adaptive Market Strength, and a true Intraday Relative Strength filter.
# - Integrated Portfolio Risk: Uses dynamic position sizing based on both
#   per-trade and total portfolio risk constraints.
# - Testable HTF Innovation: Includes a toggleable dynamic profit target that
#   adjusts based on market volatility (VIX).
#
# BUG FIX (v1.5):
# - CRITICAL FIX: Corrected a recurring `TypeError` by replacing all instances
#   of the deprecated `get_loc(..., method='ffill')` with the modern pandas
#   `get_indexer(..., method='ffill')` syntax.
# - CRITICAL BIAS FIX: Modified the VIX filter to use the PREVIOUS DAY's closing
#   VIX value, eliminating lookahead bias.
# - LOGIC FIX: Corrected the Volume Projection engine to return FALSE before the
#   first time-anchor (10 AM), ensuring all trades are validated.

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
    # 1. Volume Projection Filter
    'use_volume_projection': True,
    'volume_ma_period_daily': 20,
    'volume_multiplier_daily': 1.3,
    'volume_projection_thresholds': { 
        dt_time(10, 0): 0.20, dt_time(11, 30): 0.45,
        dt_time(13, 0): 0.65, dt_time(14, 0): 0.85
    },

    # 2. Market Strength & Slippage Filter (VIX-Adaptive)
    'use_vix_adaptive_filters': True,
    'vix_symbol': 'INDIAVIX',
    'vix_calm_threshold': 18,
    'vix_high_threshold': 25,
    'market_strength_index': 'NIFTY200_INDEX',
    'market_strength_threshold_calm': -0.25, # Stricter threshold in calm markets
    'market_strength_threshold_high': -0.75, # More lenient in volatile markets
    'base_slippage_percent': 0.05,
    'high_vol_slippage_percent': 0.15,

    # 3. Intraday Relative Strength (RS) Filter
    'use_intraday_rs_filter': True,
    
    # --- Trade Management & Profit Taking ---
    'use_partial_profit_leg': True,
    'use_dynamic_profit_target': True, # HTF-Specific Innovation
    'profit_target_rr_calm': 1.5,      # Target 1.5R in calm markets
    'profit_target_rr_high': 1.0,      # Target 1.0R in volatile markets
    'use_aggressive_breakeven': True,
    'breakeven_buffer_points': 0.05,
}

# --- HELPER FUNCTIONS ---
def get_consecutive_red_candles(df, current_loc):
    """Looks back from the candle before the setup candle to find red candles."""
    red_candles = []
    i = current_loc - 1
    while i >= 0 and (df.iloc[i]['close'] < df.iloc[i]['open']):
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def check_volume_projection(candle_time, cumulative_volume, target_daily_volume, thresholds):
    """Checks if intraday volume is on track to meet EOD targets."""
    current_time = candle_time.time()
    applicable_threshold = 0
    for threshold_time, threshold_pct in sorted(thresholds.items()):
        if current_time >= threshold_time:
            applicable_threshold = threshold_pct
        else:
            break
    # --- LOGIC FIX: Return False if before the first checkpoint ---
    if applicable_threshold == 0: 
        return False, 0, 0
    
    required_volume = target_daily_volume * applicable_threshold
    return cumulative_volume >= required_volume, cumulative_volume, required_volume

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
        self.target_list = {} # { 'YYYY-MM-DD': { 'SYMBOL': {details} } }
        self.debug_log = []
        self.tradeable_symbols = []

    def load_data(self):
        """Loads all required daily, intraday, and HTF data into memory."""
        print("Loading all data into memory...")
        try:
            self.tradeable_symbols = pd.read_csv(self.cfg['nifty_list_csv'])['Symbol'].tolist()
        except FileNotFoundError:
            print(f"FATAL ERROR: Tradeable symbols file not found at '{self.cfg['nifty_list_csv']}'. Exiting.")
            sys.exit()

        symbols_to_load = self.tradeable_symbols + [self.cfg['market_strength_index'], self.cfg['vix_symbol']]

        for symbol in symbols_to_load:
            try:
                # Load Daily
                daily_file = os.path.join(self.cfg['data_folder_base'], 'daily', f"{symbol}_daily_with_indicators.csv")
                if os.path.exists(daily_file):
                    df_d = pd.read_csv(daily_file, index_col='datetime', parse_dates=True)
                    self.daily_data[symbol] = df_d
                
                # Load Intraday
                intraday_file = os.path.join(self.cfg['intraday_data_folder'], f"{symbol}_15min.csv")
                if "NIFTY200_INDEX" in symbol: intraday_file = os.path.join(self.cfg['intraday_data_folder'], "NIFTY_200_15min.csv")
                if os.path.exists(intraday_file):
                    self.intraday_data[symbol] = pd.read_csv(intraday_file, index_col='datetime', parse_dates=True)

                # Load HTF
                htf_file = os.path.join(self.cfg['data_folder_base'], self.cfg['timeframe'], f"{symbol}_{self.cfg['timeframe']}_with_indicators.csv")
                if os.path.exists(htf_file):
                    self.htf_data[symbol] = pd.read_csv(htf_file, index_col='datetime', parse_dates=True)

            except Exception as e:
                print(f"Warning: Could not load all data for {symbol}. Error: {e}")
        print("Data loading complete.")

    def run_simulation(self):
        """Main loop to run the simulation from start to end date."""
        if not self.daily_data or self.cfg['market_strength_index'] not in self.daily_data:
            print("Error: Market strength index data not loaded. Cannot run simulation.")
            return
            
        master_dates = self.daily_data[self.cfg['market_strength_index']].loc[self.cfg['start_date']:self.cfg['end_date']].index
        
        for date in master_dates:
            progress_str = f"Processing {date.date()} | Equity: {self.portfolio['equity']:,.0f} | Positions: {len(self.portfolio['positions'])} | Targets: {len(self.target_list.get(date.strftime('%Y-%m-%d'), {}))}"
            sys.stdout.write(f"\r{progress_str.ljust(120)}"); sys.stdout.flush()

            # --- SCOUT (RUNS ON FRIDAY EOD) ---
            if date.weekday() == 4: # 4 is Friday
                self.scout_for_setups(date)

            # --- SNIPER (RUNS INTRADAY) ---
            self.sniper_monitor_and_execute(date)

            # --- EOD Portfolio Management ---
            self.manage_eod_portfolio(date)
        
        self.generate_report()

    def scout_for_setups(self, friday_date):
        """On Friday EOD, identifies valid weekly setups for the next week."""
        for symbol in self.tradeable_symbols:
            if symbol not in self.htf_data or symbol not in self.daily_data: 
                continue
            
            df_htf = self.htf_data[symbol]
            if df_htf.empty: continue

            try:
                loc_htf_indexer = df_htf.index.get_indexer([friday_date], method='ffill')
                if loc_htf_indexer[0] == -1:
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
        """Monitors intraday data for breakouts and validates with conviction engine."""
        date_str = date.strftime('%Y-%m-%d')
        todays_watchlist = self.target_list.get(date_str, {})
        if not todays_watchlist: return

        try:
            mkt_idx_intra = self.intraday_data[self.cfg['market_strength_index']].loc[date_str]
            if mkt_idx_intra.empty: return

            mkt_open = mkt_idx_intra.iloc[0]['open']
            
            # --- CRITICAL BIAS FIX: Use PREVIOUS day's VIX close ---
            vix_df = self.daily_data[self.cfg['vix_symbol']]
            # Find the integer position of the current date (or last available)
            current_date_pos_indexer = vix_df.index.get_indexer([date], method='ffill')

            # Ensure we found a date and it's not the very first day in the data
            if current_date_pos_indexer[0] < 1:
                return # Not enough history for VIX

            # Get the integer position of the previous day
            prev_day_loc = current_date_pos_indexer[0] - 1
            
            vix_close = vix_df.iloc[prev_day_loc]['close']
            if isinstance(vix_close, pd.Series):
                vix_close = vix_close.item()

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
                    if self.cfg['use_volume_projection']:
                        cum_vol = stock_intra_df.loc[:candle_time, 'volume'].sum()
                        vol_ok, _, _ = check_volume_projection(candle_time, cum_vol, details['target_daily_volume'], self.cfg['volume_projection_thresholds'])
                        if not vol_ok: continue
                    
                    if self.cfg['use_vix_adaptive_filters']:
                        mkt_strength = (mkt_candle['close'] / mkt_open - 1) * 100
                        threshold = self.cfg['market_strength_threshold_high'] if vix_close > self.cfg['vix_high_threshold'] else self.cfg['market_strength_threshold_calm']
                        if mkt_strength < threshold: continue

                    if self.cfg['use_intraday_rs_filter']:
                        stock_rs = (stock_candle['close'] / stock_open - 1) * 100
                        if stock_rs < mkt_strength: continue
                    
                    self.execute_entry(symbol, details, date, candle_time, vix_close)
                    todays_new_positions += 1
                    if symbol in todays_watchlist: del todays_watchlist[symbol]
    
    def execute_entry(self, symbol, details, date, candle_time, vix_close):
        """Calculates position size and places the trade."""
        entry_price_base = details['trigger_price']

        slippage_pct = self.cfg['base_slippage_percent']
        if self.cfg['use_vix_adaptive_filters'] and vix_close > self.cfg['vix_high_threshold']:
            slippage_pct = self.cfg['high_vol_slippage_percent']
        entry_price = entry_price_base * (1 + slippage_pct / 100)

        loc_d = self.daily_data[symbol].index.get_loc(date)
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
        """Checks for stop-loss or partial profit target hits on open positions."""
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
                self.portfolio['trades'].append({'symbol': pos['symbol'], 'pnl': (exit_price - pos['entry_price']) * shares_to_sell, 'exit_type': 'Partial Profit', **pos})
                pos['shares'] -= shares_to_sell
                pos['partial_exit'] = True
                pos['stop_loss'] = pos['entry_price']

            if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_proceeds += pos['shares'] * exit_price
                self.portfolio['trades'].append({'symbol': pos['symbol'], 'pnl': (exit_price - pos['entry_price']) * pos['shares'], 'exit_type': 'Stop-Loss', **pos})
                to_remove.append(pos_id)
        
        for pos_id in to_remove:
            self.portfolio['positions'].pop(pos_id, None)
        self.portfolio['cash'] += exit_proceeds

    def manage_eod_portfolio(self, date):
        """Updates trailing stops and equity curve at the end of the day."""
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
        """Generates and prints the final backtest summary report."""
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_folder = self.cfg['log_folder']
        summary_filename = os.path.join(log_folder, f"{timestamp}_summary_htf_advanced.txt")
        trades_filename = os.path.join(log_folder, f"{timestamp}_trades_htf_advanced.csv")
        setups_filename = os.path.join(log_folder, f"{timestamp}_setups_htf_advanced.csv")
        debug_filename = os.path.join(log_folder, f"{timestamp}_debug_log_htf_advanced.txt")

        with open(summary_filename, 'w') as f: f.write(summary_content)
        if not trades_df.empty: trades_df.to_csv(trades_filename, index=False)
        if self.all_setups_log: pd.DataFrame(self.all_setups_log).to_csv(setups_filename, index=False)
        with open(debug_filename, 'w') as f:
            for line in self.debug_log:
                f.write(f"{line}\n")
        print(f"Reports saved to '{log_folder}'")


if __name__ == '__main__':
    os.makedirs(config['log_folder'], exist_ok=True)
    
    simulator = HtfAdvancedSimulator(config)
    simulator.load_data()
    simulator.run_simulation()
