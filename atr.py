import pandas as pd
import os
import sys
import numpy as np

# SCRIPT VERSION v3.2
# v3.1: Implemented a robust merge operation for ATR calculation.
# v3.2: SAFETY FIX - Changed file handling to write to a NEW file instead of
#       overwriting the source. This prevents PermissionError issues and provides
#       better data versioning for backward compatibility.

# Fix for pandas_ta compatibility
np.NaN = np.nan
import pandas_ta as ta

# --- CONFIGURATION ---
ROOT_DIR = os.getcwd()
# INPUT: The original signals file created by the previous script.
INPUT_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_longs_data_with_signals.parquet")
# OUTPUT: The new, enhanced file that will be created.
OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_longs_data_with_signals_and_atr.parquet")
ATR_TS_PERIOD = 14

def main():
    """Loads the signals file, adds ATR, and saves it to a new file."""
    print("--- Starting Data Pipeline Enhancement: Adding ATR ---")

    if not os.path.exists(INPUT_DATA_PATH):
        print(f"ERROR: Input data file not found at {INPUT_DATA_PATH}. Please run the signal creation script first.")
        return

    print(f"Loading master data file from: {os.path.basename(INPUT_DATA_PATH)}")
    master_df = pd.read_parquet(INPUT_DATA_PATH)
    
    # Ensure datetime is the index for calculations and merging
    if 'datetime' in master_df.columns:
        master_df.set_index('datetime', inplace=True)

    print(f"Calculating ATR({ATR_TS_PERIOD}) for all symbols. This may take a moment...")

    def calculate_atr(df_group):
        return ta.atr(high=df_group['high'], low=df_group['low'], close=df_group['close'], length=ATR_TS_PERIOD)

    # Calculate ATR values using the robust .apply() method
    atr_series = master_df.groupby('symbol', observed=False).apply(calculate_atr, include_groups=False)

    # Convert the resulting Series to a DataFrame for merging
    atr_df = atr_series.to_frame(name=f'atr_{ATR_TS_PERIOD}')
    atr_df.index.names = ['symbol', 'datetime']
    atr_df.reset_index(inplace=True)

    # Merge the ATR data back into the main DataFrame
    master_df.reset_index(inplace=True)
    master_df = pd.merge(master_df, atr_df, on=['datetime', 'symbol'], how='left')
    master_df.set_index('datetime', inplace=True)

    print(f"ATR calculation complete. Saving new file to: {os.path.basename(OUTPUT_DATA_PATH)}")

    try:
        # Save the enhanced DataFrame to the NEW output path.
        master_df.to_parquet(OUTPUT_DATA_PATH)

        print(f"\n--- Success! ---")
        print(f"New file created: '{os.path.basename(OUTPUT_DATA_PATH)}'")
        print("Please update your main simulator's DATA_PATH to point to this new file.")

    except Exception as e:
        print(f"\nERROR: An error occurred while saving the file: {e}")

if __name__ == "__main__":
    main()

