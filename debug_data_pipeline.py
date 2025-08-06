# debug_data_pipeline.py
#
# Description:
# A forensic analysis tool to debug data integrity issues in the TFL pipeline.
# This script isolates a single symbol and setup date to trace the data from
# raw daily files, through monthly resampling, to the final trigger price
# calculation, comparing it against the execution data to find discrepancies.
#
# How to use:
# 1. Set the parameters in the CONFIG section below.
# 2. Run the script: python debug_data_pipeline.py
# 3. Analyze the output to find the source of the data mismatch.

import pandas as pd
import os

# --- CONFIGURATION ---
# Set the parameters for the specific trade you want to investigate
CONFIG = {
    'symbol': 'CIPLA',
    'setup_year': 2020,
    'setup_month': 3, # March
    'execution_month': 4, # April
    'raw_data_folder': os.path.join("data", "universal_historical_data"),
}

def get_consecutive_red_candles_debug(df, current_loc):
    """A debug version of the red candle identification logic."""
    red_candles = []
    i = current_loc - 1
    # A monthly candle is red if its close is less than its open
    while i >= 0 and (df.iloc[i]['close'] < df.iloc[i]['open']):
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def main():
    """Main function to run the forensic analysis."""
    print("--- Starting Data Pipeline Forensic Analysis ---")
    
    # --- 1. Load Raw Daily Data ---
    symbol = CONFIG['symbol']
    raw_file = os.path.join(CONFIG['raw_data_folder'], f"{symbol}_daily.csv")
    
    if not os.path.exists(raw_file):
        print(f"ERROR: Raw data file not found at: {raw_file}")
        return
        
    try:
        df_daily_raw = pd.read_csv(raw_file, index_col='datetime', parse_dates=True)
        print(f"Successfully loaded raw daily data for {symbol}.")
    except Exception as e:
        print(f"ERROR: Could not load raw data file. Reason: {e}")
        return

    # --- 2. Replicate Monthly Resampling ---
    # This logic is copied directly from universal_calculate_indicators.py
    aggregation_rules = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }
    df_monthly_resampled = df_daily_raw.resample('MS').agg(aggregation_rules).dropna()
    print("\nReplicated monthly resampling logic.")

    # --- 3. Isolate the Setup Candle and Preceding Candles ---
    setup_date_str = f"{CONFIG['setup_year']}-{CONFIG['setup_month']}-01"
    setup_datetime = pd.to_datetime(setup_date_str)
    
    try:
        loc_monthly = df_monthly_resampled.index.get_loc(setup_datetime)
        setup_candle = df_monthly_resampled.iloc[loc_monthly]
    except KeyError:
        print(f"ERROR: No monthly data found for {setup_datetime.strftime('%Y-%m')}. Cannot proceed.")
        return

    print(f"\n--- Analysis of Setup Month: {setup_datetime.strftime('%B %Y')} ---")
    print("Monthly Setup Candle Data:")
    print(setup_candle)

    # Replicate the logic to find the preceding red candles
    red_candles = get_consecutive_red_candles_debug(df_monthly_resampled, loc_monthly)

    if not red_candles:
        print("\nWARNING: No preceding red monthly candles were found before the setup candle.")
    else:
        print(f"\nFound {len(red_candles)} preceding red monthly candle(s).")
        for i, candle in enumerate(red_candles):
            print(f"  Red Candle #{i+1} ({candle.name.strftime('%Y-%m')}): High = {candle['high']}")

    # --- 4. Replicate Trigger Price Calculation ---
    # This logic is copied directly from monthly_tfl_simulator.py
    if not red_candles:
         # If there are no red candles, the trigger is just the high of the setup candle
        trigger_price = setup_candle['high']
    else:
        trigger_price = max([c['high'] for c in red_candles] + [setup_candle['high']])

    print(f"\nCalculated Trigger Price based on monthly data: {trigger_price:.2f}")

    # --- 5. Compare with Execution Month Daily Data ---
    execution_start = f"{CONFIG['setup_year']}-{CONFIG['execution_month']}-01"
    execution_end = pd.to_datetime(execution_start) + pd.offsets.MonthEnd(1)
    
    df_execution_days = df_daily_raw.loc[execution_start:execution_end]

    print(f"\n--- Verification Against Execution Month: {pd.to_datetime(execution_start).strftime('%B %Y')} ---")
    print("Daily High Prices vs. Calculated Trigger Price:")
    
    trade_possible = False
    for date, candle in df_execution_days.iterrows():
        daily_high = candle['high']
        possible_str = "POSSIBLE" if daily_high >= trigger_price else "IMPOSSIBLE"
        if daily_high >= trigger_price:
            trade_possible = True
        print(f"  {date.strftime('%Y-%m-%d')}: Daily High = {daily_high:.2f} | Trigger Price = {trigger_price:.2f} -> Trade is {possible_str}")

    print("\n--- CONCLUSION ---")
    if trade_possible:
        print("The calculated trigger price WAS reachable during the execution month.")
        print("The issue may lie elsewhere in the simulator's intraday logic or state management.")
    else:
        print("CRITICAL FINDING: The calculated trigger price was NEVER reached during the execution month.")
        print("This confirms a data misalignment issue. The monthly resampled 'high' does not reflect the true daily highs.")
        print("Next Step: Manually inspect the raw daily data for the setup month to see why the resampled 'high' is incorrect.")

if __name__ == "__main__":
    main()
