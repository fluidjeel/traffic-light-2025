# daily_entry_scanner.py
#
# Description:
# A streamlined backtest script focused solely on identifying and logging all
# valid entry signals for the daily TFL strategy.
#
# MODIFICATION (v1.5 - MACD Histogram Range Filter):
# 1. ADDED: A new configuration option, 'macd_hist_range', to filter signals
#    based on a specific range of MACD Histogram values.
# 2. MODIFIED: The trend/momentum filter logic to use this new range check.

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
        'universal_intraday_folder': os.path.join('data', 'universal_historical_data'),
    },

    # --- Logging & Strategy Info ---
    'log_folder': 'backtest_logs',
    'strategy_name': 'daily_entry_scanner',
    'output_filename': 'daily_entry_signals.csv',

    # --- Core Daily Strategy Parameters (EOD Scan) ---
    'use_ema_filter': True,
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
    
    # NEW MACD RANGE FILTER:
    'use_macd_range_filter': False,
    'macd_hist_range': (2.0, 5.0), # Example range to test
    
    'use_trend_momentum_filter': False, # This is now deprecated, using macd_hist_range instead
    'ema_fast_period': 20,
    'ema_slow_period': 50,
    
    # --- VIX Settings for Context ---
    'vix_symbol': 'INDIAVIX',
}

def get_consecutive_red_candles(df, current_loc):
    """Identifies consecutive red candles prior to a given location."""
    red_candles = []
    i = current_loc - 1
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def run_scanner(cfg):
    """
    Main function to run the backtest and generate the list of entry signals.
    """
    start_time = time.time()
    
    cfg['data_folder_base'] = cfg['data_pipeline_config']['universal_processed_folder']
    cfg['intraday_data_folder'] = cfg['data_pipeline_config']['universal_intraday_folder']
    daily_folder = os.path.join(cfg['data_folder_base'], 'daily')
    intraday_folder = cfg['intraday_data_folder']
    
    strategy_log_folder = os.path.join(cfg['log_folder'], cfg['strategy_name'])
    os.makedirs(strategy_log_folder, exist_ok=True)

    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return

    print("Loading all Daily and 15-min data into memory...")
    daily_data, intraday_data = {}, {}
    symbols_to_load = symbols + [cfg['regime_index_symbol'], cfg['vix_symbol']]
    for symbol in symbols_to_load:
        try:
            daily_file = os.path.join(daily_folder, f"{symbol}_daily_with_indicators.csv")
            if os.path.exists(daily_file):
                df_d = pd.read_csv(daily_file, index_col='datetime', parse_dates=True)
                df_d.rename(columns=lambda x: x.lower().replace('.', '_'), inplace=True)
                df_d['red_candle'] = df_d['close'] < df_d['open']
                df_d['green_candle'] = df_d['close'] > df_d['open']
                daily_data[symbol] = df_d
            
            intraday_file = os.path.join(intraday_folder, f"{symbol}_15min.csv")
            if os.path.exists(intraday_file):
                intraday_data[symbol] = pd.read_csv(intraday_file, index_col='datetime', parse_dates=True)
        except Exception as e:
            print(f"Warning: Could not load data for {symbol}. Error: {e}")
    print("Data loading complete.")

    watchlist = {}
    entry_signals = []
    
    master_dates = daily_data[cfg['regime_index_symbol']].loc[cfg['start_date']:cfg['end_date']].index

    print("Starting Daily Entry Scanner...")
    for date in master_dates:
        progress_str = f"Scanning {date.date()} | Signals Found: {len(entry_signals)}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()

        todays_watchlist = watchlist.get(date, {})
        
        if todays_watchlist:
            try:
                vix_df = daily_data[cfg['vix_symbol']]
                vix_close_t1 = vix_df.loc[:date].iloc[-2]['close']
                today_intraday_candles = intraday_data.get(cfg['regime_index_symbol'], pd.DataFrame()).loc[date.date().strftime('%Y-%m-%d')]
            except (KeyError, IndexError): 
                continue

            if not today_intraday_candles.empty:
                for candle_time, _ in today_intraday_candles.iterrows():
                    for symbol, details in list(todays_watchlist.items()):
                        if symbol not in intraday_data or candle_time not in intraday_data[symbol].index:
                            continue
                        
                        candle = intraday_data[symbol].loc[candle_time]
                        
                        if candle['high'] >= details['trigger_price']:
                            entry_signals.append({
                                'setup_date': details['setup_date'],
                                'symbol': symbol,
                                'entry_timestamp': candle_time,
                                'trigger_price': details['trigger_price'],
                                'vix_on_entry_day': vix_close_t1,
                            })
                            del todays_watchlist[symbol]

        market_uptrend = True
        if cfg['market_regime_filter']:
            if date in daily_data[cfg['regime_index_symbol']].index:
                index_data = daily_data[cfg['regime_index_symbol']].loc[date]
                if cfg['regime_ema_cross']:
                    # Use EMA cross-over for a more nuanced trend filter
                    fast_period, slow_period = cfg['regime_ema_cross']
                    if index_data[f"ema_{fast_period}"] < index_data[f"ema_{slow_period}"]:
                        market_uptrend = False
                elif index_data['close'] < index_data[f"ema_{cfg['regime_ma_period']}"]:
                    # Default behavior: check against a single EMA
                    market_uptrend = False
            else:
                market_uptrend = False
        
        if market_uptrend:
            for symbol in symbols:
                if symbol not in daily_data or date not in daily_data[symbol].index:
                    continue
                df = daily_data[symbol]
                try:
                    loc = df.index.get_loc(date)
                    if loc < 2: continue
                    setup_candle = df.iloc[loc]
                    
                    # Core Price Action: Must be a green candle to resume an uptrend
                    if not setup_candle['green_candle']:
                        continue
                    
                    # NEW: EMA Filter is now conditional based on the config flag
                    if cfg['use_ema_filter'] and not (setup_candle['close'] > setup_candle[f"ema_{cfg['ema_period']}"]):
                        continue
                    
                    red_candles = get_consecutive_red_candles(df, loc)
                    if not red_candles: continue

                    if cfg['volume_filter']:
                        if pd.isna(setup_candle[f"volume_{cfg['volume_ma_period']}_sma"]) or \
                           setup_candle['volume'] < (setup_candle[f"volume_{cfg['volume_ma_period']}_sma"] * cfg['volume_multiplier']):
                            continue
                    
                    if cfg['turnover_filter']:
                        if 'turnover_20_sma' not in setup_candle.index or pd.isna(setup_candle['turnover_20_sma']) or \
                           setup_candle['turnover_20_sma'] < cfg['min_avg_turnover']:
                            continue
                    
                    # NEW MACD RANGE FILTER: Check if macdh is within the specified range
                    if cfg['use_macd_range_filter']:
                        macd_hist = setup_candle.get('macdh_12_26_9')
                        if pd.isna(macd_hist) or not (cfg['macd_hist_range'][0] <= macd_hist <= cfg['macd_hist_range'][1]):
                            continue
                    
                    if cfg['use_trend_momentum_filter']:
                        ema_fast = setup_candle.get(f"ema_{cfg['ema_fast_period']}")
                        ema_slow = setup_candle.get(f"ema_{ema_slow_period}")
                        if pd.isna(ema_fast) or pd.isna(ema_slow) or ema_fast <= ema_slow:
                            continue
                            
                        macd_hist = setup_candle.get('macdh_12_26_9')
                        if pd.isna(macd_hist) or macd_hist <= 0:
                            continue
                    
                    trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])
                    next_day = date + timedelta(days=1)
                    if next_day in master_dates:
                        if next_day not in watchlist: watchlist[next_day] = {}
                        watchlist[next_day][symbol] = {'trigger_price': trigger_price, 'setup_date': date}
                except (KeyError, IndexError):
                    continue

    print("\n\n--- SCAN COMPLETE ---")
    
    if not entry_signals:
        print("No entry signals were found for the given period and configuration.")
        return

    entry_signals_df = pd.DataFrame(entry_signals)
    output_path = os.path.join(strategy_log_folder, cfg['output_filename'])
    entry_signals_df.to_csv(output_path, index=False)

    print(f"Successfully found {len(entry_signals_df)} entry signals.")
    print(f"Results saved to: {output_path}")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_scanner(config)
