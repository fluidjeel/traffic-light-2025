# weekly_signal_characterizer.py
#
# Description:
# An analysis script that separates weekly entry signals into "Winners" and "Losers"
# and then analyzes the characteristics of the weekly setup for each group.
# It calculates the average value of a comprehensive suite of indicators for
# both groups to find statistical differences for building new filters.

import pandas as pd
import os
import sys
import numpy as np
from collections import Counter

# --- CONFIGURATION ---
config = {
    'log_folder': 'backtest_logs',
    'scanner_strategy_name': 'weekly_entry_scanner',
    'scanner_output_filename': 'weekly_entry_signals.csv',
    'processed_data_folder': os.path.join('data', 'universal_processed', 'weekly'), # Use weekly processed data
    'index_symbol': 'NIFTY200_INDEX',

    'stop_loss_lookback_weeks': 2, # Stop-loss lookback based on weekly candles
    'max_holding_weeks': 10, # Max weeks to hold for outcome determination
    'win_threshold_r': 1.0, # A signal is a "Winner" if it achieves this R multiple
}

def get_signal_outcome(signal, weekly_data, cfg):
    """Determines if a weekly signal was a winner or a loser."""
    symbol = signal['symbol']
    entry_price = signal['trigger_price']
    setup_week_date = signal['setup_week_date']
    entry_date = signal['entry_date'] # The daily date the trigger was hit

    if symbol not in weekly_data: return None

    df_weekly = weekly_data[symbol]

    # --- Calculate Initial Stop-Loss based on weekly candles ---
    try:
        setup_week_loc_indexer = df_weekly.index.get_indexer([setup_week_date], method='ffill')
        if setup_week_loc_indexer[0] == -1: return None
        setup_week_loc = setup_week_loc_indexer[0]
        
        stop_loss_slice = df_weekly.iloc[max(0, setup_week_loc - cfg['stop_loss_lookback_weeks']) : setup_week_loc + 1]
        
        if stop_loss_slice.empty: return None
        
        stop_loss_price = stop_loss_slice['low'].min()
    except (KeyError, IndexError): return None

    initial_risk = entry_price - stop_loss_price
    if initial_risk <= 0: return None

    win_target_price = entry_price + (initial_risk * cfg['win_threshold_r'])
    
    # --- Trace Weekly Price Action from the entry_date ---
    trade_horizon_start_loc_indexer = df_weekly.index.get_indexer([entry_date], method='bfill')
    if trade_horizon_start_loc_indexer[0] == -1: return None
    trade_horizon_start_loc = trade_horizon_start_loc_indexer[0]

    trade_horizon = df_weekly.iloc[trade_horizon_start_loc : trade_horizon_start_loc + cfg['max_holding_weeks']]
    
    for _, candle in trade_horizon.iterrows():
        if candle['low'] <= stop_loss_price:
            return 'Loser' # Hit stop-loss
        if candle['high'] >= win_target_price:
            return 'Winner' # Hit profit target
    
    return 'Loser' # Timed out or did not hit target

def main():
    """Main function to run the weekly characterization analysis."""
    print("--- Starting Weekly Signal Characterization Analysis ---")

    cfg = config
    signals_filepath = os.path.join(cfg['log_folder'], cfg['scanner_strategy_name'], cfg['scanner_output_filename'])
    if not os.path.exists(signals_filepath):
        print(f"ERROR: Signals file not found. Run scanner first."); return
        
    signals_df = pd.read_csv(signals_filepath, parse_dates=['setup_week_date', 'entry_date'])
    print(f"Loaded {len(signals_df)} entry signals.")

    print("Loading historical data...")
    weekly_data = {}
    symbols_to_load = signals_df['symbol'].unique().tolist() + [config['index_symbol']]
    for symbol in symbols_to_load:
        try:
            weekly_file = os.path.join(cfg['processed_data_folder'], f"{symbol}_weekly_with_indicators.csv")
            if os.path.exists(weekly_file):
                df = pd.read_csv(weekly_file, index_col='datetime', parse_dates=True)
                df.rename(columns=lambda x: x.lower().replace('.', '_'), inplace=True)
                weekly_data[symbol] = df
        except Exception as e:
            print(f"Warning: Could not load data for {symbol}. Error: {e}")
    print("Data loading complete.")

    print("Classifying signals as Winners vs. Losers...")
    signals_df['outcome'] = signals_df.apply(get_signal_outcome, axis=1, weekly_data=weekly_data, cfg=cfg)

    winners = signals_df[signals_df['outcome'] == 'Winner'].copy()
    losers = signals_df[signals_df['outcome'] == 'Loser'].copy()

    if winners.empty or losers.empty:
        print("\nCould not find both winners and losers to compare. Analysis cannot proceed.")
        return

    print(f"Found {len(winners)} Winners and {len(losers)} Losers.")

    # --- Analyze Characteristics ---
    characteristics = []
    # These indicators should be present in your universal_calculate_indicators.py output for weekly timeframe
    indicators_to_check = [
        'ema_8', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
        'rsi_14', 'macd_12_26_9', 'macdh_12_26_9', 'macds_12_26_9',
        'stochk_14_3_3', 'stochd_14_3_3', 'return_30',
        'atr_14_pct', 'bbu_20_2_0', 'bbm_20_2_0', 'bbl_20_2_0',
        'obv', 'volume_20_sma', 'volume_50_sma', 'volume',
        'prox_52w_high', 'body_ratio'
    ]
    
    index_df = weekly_data[config['index_symbol']]

    for group_name, group_df in [('Winners', winners), ('Losers', losers)]:
        char_values = {indicator: [] for indicator in indicators_to_check + ['rs_vs_index']}
        
        for _, signal in group_df.iterrows():
            symbol = signal['symbol']
            setup_week_date = signal['setup_week_date'] # Use the weekly setup date for characteristics

            if symbol in weekly_data:
                df_weekly = weekly_data[symbol]
                try:
                    # Find the corresponding weekly candle for the setup_week_date
                    indexer = df_weekly.index.get_indexer([setup_week_date], method='ffill')
                    if indexer[0] == -1:
                        continue
                    
                    setup_candle_weekly = df_weekly.iloc[indexer[0]]
                    
                    for indicator in indicators_to_check:
                        if indicator in setup_candle_weekly and pd.notna(setup_candle_weekly[indicator]):
                            char_values[indicator].append(setup_candle_weekly[indicator])
                    
                    # Calculate RS vs Index using weekly returns
                    actual_setup_ts = setup_candle_weekly.name 
                    if actual_setup_ts in index_df.index and pd.notna(setup_candle_weekly['return_30']) and pd.notna(index_df.loc[actual_setup_ts, 'return_30']):
                        rs = setup_candle_weekly['return_30'] - index_df.loc[actual_setup_ts, 'return_30']
                        char_values['rs_vs_index'].append(rs)
                except (KeyError, IndexError):
                    continue

        avg_chars = {f"Avg {key}": np.mean(val) for key, val in char_values.items() if val}
        avg_chars['Group'] = group_name
        avg_chars['Count'] = len(group_df)
        characteristics.append(avg_chars)

    # --- Print Comparison Report ---
    print("\n--- Winner vs. Loser Characteristic Analysis (Weekly Setup Day) ---")
    report_df = pd.DataFrame(characteristics).set_index('Group')
    
    for col in report_df.columns:
        if 'pct' in col or 'prox' in col or 'return' in col or 'rs_vs' in col:
            report_df[col] = report_df[col].map('{:,.2f}%'.format)
        elif 'volume' in col or 'obv' in col:
            report_df[col] = report_df[col].map('{:,.0f}'.format)
        elif col != 'Count':
            report_df[col] = report_df[col].map('{:,.2f}'.format)
            
    print(report_df.T)
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()
