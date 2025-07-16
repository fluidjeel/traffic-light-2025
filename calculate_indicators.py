# calculate_indicators.py
#
# Description:
# This script reads the daily, weekly, and monthly historical data,
# calculates common technical indicators (like EMA, RSI, MACD),
# and saves the enhanced data into new files.
#
# Prerequisites:
# 1. You must first run 'fyers_equity_scraper.py' to download daily data.
# 2. You must then run 'resample_data.py' to generate weekly and monthly data.
# 3. Required libraries installed: pip install pandas pandas-ta
#
# How to use:
# 1. Make sure the 'historical_data', 'weekly_data', and 'monthly_data' folders
#    exist and contain the data files.
# 2. Run this script. It will create three new folders:
#    - 'daily_with_indicators'
#    - 'weekly_with_indicators'
#    - 'monthly_with_indicators'
#    These folders will contain the original data plus the calculated indicators.

import os
import sys
import pandas as pd
import pandas_ta as ta

def calculate_and_save_indicators(input_dir, output_dir):
    """
    Reads all CSV files from an input directory, calculates technical indicators,
    and saves the results to an output directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Warning: Input directory '{input_dir}' not found. Skipping.")
        return

    # Get a list of all data files in the input directory
    try:
        data_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    except Exception as e:
        print(f"Error reading from input directory '{input_dir}': {e}")
        return

    if not data_files:
        print(f"No data files found in '{input_dir}'. Skipping.")
        return

    total_files = len(data_files)
    print(f"\n--- Processing directory: '{input_dir}' ({total_files} files) ---")

    # Loop through each file and process it
    for i, filename in enumerate(data_files):
        symbol_name = filename.split('_')[0]
        print(f"  > Calculating for {symbol_name} ({i+1}/{total_files})...", end="")
        
        try:
            # Construct full file path
            file_path = os.path.join(input_dir, filename)
            
            # Read the data into a DataFrame
            df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)

            if df.empty:
                print(" [Skipped: Empty file]")
                continue

            # --- Calculate Technical Indicators using pandas-ta ---
            # You can customize this section by adding or removing indicators.
            # For a full list of available indicators, see the pandas-ta documentation.
            
            # Exponential Moving Averages (EMA)
            df.ta.ema(length=10, append=True)
            df.ta.ema(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)

            # Relative Strength Index (RSI)
            df.ta.rsi(length=14, append=True)
            
            # Moving Average Convergence Divergence (MACD)
            # This will add 3 columns: MACD, MACD_histogram, MACD_signal
            df.ta.macd(fast=12, slow=26, signal=9, append=True)

            # Bollinger Bands (BBANDS)
            # This will add 5 columns for upper, middle, lower bands, bandwidth, and percent.
            df.ta.bbands(length=20, std=2, append=True)

            # --- Save the enhanced data ---
            output_path = os.path.join(output_dir, filename.replace('.csv', '_with_indicators.csv'))
            df.to_csv(output_path)
            print(" [Done]")

        except Exception as e:
            print(f" [Error: {e}]")

    print(f"--- Finished processing '{input_dir}' ---")

if __name__ == "__main__":
    print("Starting technical indicator calculation process...")
    
    # Process daily data
    calculate_and_save_indicators("historical_data", "daily_with_indicators")
    
    # Process weekly data
    calculate_and_save_indicators("weekly_data", "weekly_with_indicators")

    # Process monthly data
    calculate_and_save_indicators("monthly_data", "monthly_with_indicators")

    print("\n--- All calculations complete! ---")
