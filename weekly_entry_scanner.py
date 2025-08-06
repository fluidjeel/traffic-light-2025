# weekly_entry_scanner.py
#
# Description:
# A streamlined script focused solely on identifying and logging all
# valid entry signals for the weekly TFL strategy. It identifies setups
# based on weekly candle patterns and then simulates the daily monitoring
# for the trigger price to log the actual entry date and VIX value.
#
# MODIFICATION (v2.5 - Refined Filters for Quality after Volume Increase):
# 1. DISABLED: 'use_max_prox_52w_high_filter' confirmed as False.
# 2. UPDATED: 'min_return_30' and 'min_rs_vs_index' for tighter quality control.
# 3. UPDATED: 'macd_signal_range' for tighter quality control.

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
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',

    # --- DATA PIPELINE CONFIGURATION ---
    'data_pipeline_config': {
        'use_universal_pipeline': True,
        'universal_processed_folder': os.path.join('data', 'universal_processed'),
        'universal_intraday_folder': os.path.join('data', 'universal_historical_data'), # Needed for VIX and daily data
    },

    # --- Logging & Strategy Info ---
    'log_folder': 'backtest_logs',
    'strategy_name': 'weekly_entry_scanner',
    'output_filename': 'weekly_entry_signals.csv',

    # --- Core Weekly Strategy Parameters (EOD Scan) ---
    'timeframe': 'weekly', # Specify the timeframe for this scanner
    'use_ema_filter': False,
    'ema_period': 20,
    'market_regime_filter': False,
    'regime_index_symbol': 'NIFTY200_INDEX',
    'regime_ma_period': 50,
    'regime_ema_cross': None,
    'volume_filter': False,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    'turnover_filter': False,
    'min_avg_turnover': 50000000,
    'use_macd_range_filter': True,
    'macd_hist_range': (0.0, 4.0), # Keep as is, differentiation is low

    # REFINED FILTERS BASED ON LATEST CHARACTERISTICS ANALYSIS
    'use_macd_signal_range_filter': True,
    'macd_signal_range': (3.0, 15.0), # Tighter upper bound for quality

    'use_min_return_30_filter': True,
    'min_return_30': 48.0, # Tighter minimum for quality

    'use_max_prox_52w_high_filter': False, # Confirmed DISABLED for trade count
    'max_prox_52w_high': -15.0, # Value doesn't matter if disabled

    'use_min_rs_vs_index_filter': True,
    'min_rs_vs_index': 28.0, # Tighter minimum for quality
    
    'use_min_rsi_filter': False, # Keep disabled
    'min_rsi': 60.0, # Value doesn't matter if disabled
    
    # --- VIX Settings for Context (from daily data) ---
    'vix_symbol': 'INDIAVIX',
}

def get_consecutive_red_candles(df, current_loc):
    """Identifies consecutive red candles prior to a given location."""
    red_candles = []
    i = current_loc - 1
    # Check for red candle (close < open)
    while i >= 0 and df.iloc[i]['close'] < df.iloc[i]['open']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def run_scanner(cfg):
    """
    Main function to run the weekly scanner and generate the list of entry signals.
    """
    start_time = time.time()
    
    # Use the processed folder for the specified timeframe (e.g., 'weekly')
    weekly_processed_folder = os.path.join(cfg['data_pipeline_config']['universal_processed_folder'], cfg['timeframe'])
    daily_processed_folder = os.path.join(cfg['data_pipeline_config']['universal_processed_folder'], 'daily') # For VIX and daily data
    
    strategy_log_folder = os.path.join(cfg['log_folder'], cfg['strategy_name'])
    os.makedirs(strategy_log_folder, exist_ok=True)

    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return

    print(f"Loading all {cfg['timeframe'].capitalize()} and Daily data into memory...")
    weekly_data = {} # Use weekly_data for higher timeframe (weekly)
    daily_data = {} # To get daily VIX and monitor daily triggers
    symbols_to_load = symbols + [cfg['regime_index_symbol'], cfg['vix_symbol']]
    
    data_loaded_count = 0 
    for symbol in symbols_to_load:
        try:
            weekly_file = os.path.join(weekly_processed_folder, f"{symbol}_{cfg['timeframe']}_with_indicators.csv")
            if os.path.exists(weekly_file):
                df_w = pd.read_csv(weekly_file, index_col='datetime', parse_dates=True)
                df_w.rename(columns=lambda x: x.lower().replace('.', '_'), inplace=True)
                df_w['red_candle'] = df_w['close'] < df_w['open']
                df_w['green_candle'] = df_w['close'] > df_w['open']
                weekly_data[symbol] = df_w
                data_loaded_count += 1

            daily_file = os.path.join(daily_processed_folder, f"{symbol}_daily_with_indicators.csv")
            if os.path.exists(daily_file):
                df_d = pd.read_csv(daily_file, index_col='datetime', parse_dates=True)
                df_d.rename(columns=lambda x: x.lower().replace('.', '_'), inplace=True)
                daily_data[symbol] = df_d
                data_loaded_count += 1

        except Exception as e:
            print(f"Warning: Could not load data for {symbol}. Error: {e}")
    print(f"Data loading complete. Loaded data for {data_loaded_count} files.")
    if data_loaded_count == 0:
        print("CRITICAL: No data files were loaded. Please check data paths and file existence.")
        return


    weekly_setups_watchlist = {} # Stores weekly setups to be monitored daily
    entry_signals = []
    
    if cfg['regime_index_symbol'] not in daily_data:
        print(f"Error: Daily data for {cfg['regime_index_symbol']} not loaded. Cannot proceed.")
        return
    master_daily_dates = daily_data[cfg['regime_index_symbol']].loc[cfg['start_date']:cfg['end_date']].index
    if master_daily_dates.empty:
        print("CRITICAL: Master daily dates range is empty. Check start/end dates or index data.")
        return

    print(f"Starting {cfg['timeframe'].capitalize()} Entry Scanner...")

    for current_daily_date in master_daily_dates:
        progress_str = f"Scanning {current_daily_date.date()} | Weekly Signals Found: {len(entry_signals)}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()

        # --- Phase 1: Monitor for entries from previously identified weekly setups ---
        vix_on_entry_day = np.nan
        if cfg['vix_symbol'] in daily_data and current_daily_date in daily_data[cfg['vix_symbol']].index:
            vix_on_entry_day = daily_data[cfg['vix_symbol']].loc[current_daily_date]['close']
        
        symbols_to_monitor_today = list(weekly_setups_watchlist.keys())
        for symbol in symbols_to_monitor_today:
            if symbol not in daily_data or current_daily_date not in daily_data[symbol].index:
                continue

            daily_candle = daily_data[symbol].loc[current_daily_date]
            setup_info = weekly_setups_watchlist[symbol]

            if daily_candle['high'] >= setup_info['trigger_price']:
                entry_signals.append({
                    'setup_week_date': setup_info['setup_week_date'],
                    'symbol': symbol,
                    'trigger_price': setup_info['trigger_price'],
                    'entry_date': current_daily_date,
                    'vix_on_entry_day': vix_on_entry_day,
                })
                del weekly_setups_watchlist[symbol]
        
        # --- Phase 2: Identify new weekly setups on Fridays ---
        if current_daily_date.weekday() == 4:
            market_uptrend = True
            if cfg['market_regime_filter']:
                if cfg['regime_index_symbol'] in weekly_data and current_daily_date in weekly_data[cfg['regime_index_symbol']].index:
                    index_data_weekly = weekly_data[cfg['regime_index_symbol']].loc[current_daily_date]
                    if cfg['regime_ema_cross']:
                        fast_period, slow_period = cfg['regime_ema_cross']
                        if index_data_weekly[f"ema_{fast_period}"] < index_data_weekly[f"ema_{slow_period}"]:
                            market_uptrend = False
                    elif index_data_weekly['close'] < index_data_weekly[f"ema_{cfg['regime_ma_period']}"]:
                        market_uptrend = False
                else:
                    market_uptrend = False

            if not market_uptrend:
                continue

            for symbol in symbols:
                if symbol not in weekly_data:
                    continue

                df_weekly = weekly_data[symbol]
                try:
                    loc_indexer = df_weekly.index.get_indexer([current_daily_date], method='ffill')
                    if not loc_indexer.size or loc_indexer[0] == -1:
                        continue
                    
                    loc = loc_indexer[0]
                    setup_candle_weekly = df_weekly.iloc[loc]
                    
                    if loc < 2:
                        continue
                    
                    if not setup_candle_weekly['green_candle']:
                        continue
                    
                    ema_col_name = f"ema_{cfg['ema_period']}"
                    if cfg['use_ema_filter'] and not (setup_candle_weekly['close'] > setup_candle_weekly[ema_col_name]):
                        continue
                    
                    red_candles_weekly = get_consecutive_red_candles(df_weekly, loc)
                    if not red_candles_weekly:
                        continue

                    if cfg['volume_filter']:
                        volume_sma_col_name = f"volume_{cfg['volume_ma_period']}_sma"
                        if pd.isna(setup_candle_weekly[volume_sma_col_name]) or \
                           setup_candle_weekly['volume'] < (setup_candle_weekly[volume_sma_col_name] * cfg['volume_multiplier']):
                            continue
                    
                    if cfg['turnover_filter']:
                        if 'turnover_20_sma' not in setup_candle_weekly.index or pd.isna(setup_candle_weekly['turnover_20_sma']) or \
                           setup_candle_weekly['turnover_20_sma'] < cfg['min_avg_turnover']:
                            continue
                        
                    if cfg['use_macd_range_filter']:
                        macd_hist = setup_candle_weekly.get('macdh_12_26_9')
                        if pd.isna(macd_hist) or not (cfg['macd_hist_range'][0] <= macd_hist <= cfg['macd_hist_range'][1]):
                            continue

                    # NEW FILTER: MACD Signal Line Range
                    if cfg['use_macd_signal_range_filter']:
                        macd_signal = setup_candle_weekly.get('macds_12_26_9')
                        if pd.isna(macd_signal) or not (cfg['macd_signal_range'][0] <= macd_signal <= cfg['macd_signal_range'][1]):
                            continue

                    # NEW FILTER: Minimum 30-period Return
                    if cfg['use_min_return_30_filter']:
                        return_30 = setup_candle_weekly.get('return_30')
                        if pd.isna(return_30) or return_30 < cfg['min_return_30']:
                            continue

                    # NEW FILTER: Maximum Proximity to 52-Week High
                    if cfg['use_max_prox_52w_high_filter']:
                        prox_52w_high = setup_candle_weekly.get('prox_52w_high')
                        # Filter out if prox_52w_high is GREATER than the max (i.e., too close to high)
                        if pd.isna(prox_52w_high) or prox_52w_high > cfg['max_prox_52w_high']:
                            continue

                    # NEW FILTER: Minimum Relative Strength vs. Index
                    if cfg['use_min_rs_vs_index_filter']:
                        # Need to get index_df from weekly_data and calculate rs here
                        if cfg['regime_index_symbol'] not in weekly_data or setup_candle_weekly.name not in weekly_data[cfg['regime_index_symbol']].index:
                            continue # Cannot calculate RS if index data is missing for this date
                        index_return_30 = weekly_data[cfg['regime_index_symbol']].loc[setup_candle_weekly.name].get('return_30')
                        stock_return_30 = setup_candle_weekly.get('return_30')

                        if pd.isna(stock_return_30) or pd.isna(index_return_30):
                            continue # Cannot calculate RS if either return is missing

                        rs = stock_return_30 - index_return_30
                        if rs < cfg['min_rs_vs_index']:
                            continue
                    
                    # NEW FILTER: Minimum RSI
                    if cfg['use_min_rsi_filter']:
                        rsi_14 = setup_candle_weekly.get('rsi_14')
                        if pd.isna(rsi_14) or rsi_14 < cfg['min_rsi']:
                            continue
                        
                    trigger_price = max([c['high'] for c in red_candles_weekly] + [setup_candle_weekly['high']])
                    
                    weekly_setups_watchlist[symbol] = {
                        'setup_week_date': current_daily_date,
                        'trigger_price': trigger_price,
                    }
                except (KeyError, IndexError) as e:
                    continue

    print("\n\n--- WEEKLY SCAN COMPLETE ---")
    
    if not entry_signals:
        print("No weekly entry signals were found for the given period and configuration.")
        return

    entry_signals_df = pd.DataFrame(entry_signals)
    output_path = os.path.join(strategy_log_folder, cfg['output_filename'])
    entry_signals_df.to_csv(output_path, index=False)

    print(f"Successfully found {len(entry_signals_df)} weekly entry signals.")
    print(f"Results saved to: {output_path}")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_scanner(config)