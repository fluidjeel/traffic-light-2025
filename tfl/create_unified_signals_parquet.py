import pandas as pd
import os
import sys

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# This script should be run from the root of your project (e.g., D:\algo-2025)
ROOT_DIR = os.getcwd()
DATA_DIRECTORY = os.path.join(ROOT_DIR, "data", "strategy_specific_data")

# --- Input Files ---
# These are the final, corrected data files from your long and short pipelines.
LONGS_DATA_PATH = os.path.join(DATA_DIRECTORY, "tfl_longs_data_with_signals_and_atr.parquet")
SHORTS_DATA_PATH = os.path.join(DATA_DIRECTORY, "tfl_shorts_data_with_signals_and_atr.parquet")

# --- Output File ---
# This is the new, master data file that the v2.0 unified simulator will use.
OUTPUT_FILENAME = "tfl_unified_data.parquet"

# ==============================================================================
# --- SCRIPT LOGIC ---
# ==============================================================================

def main():
    """
    Loads the separate long and short signal files, adds a direction flag,
    and combines them into a single, unified master data file.
    """
    print("--- TFL Unified Data File Creation Utility ---")

    # --- Load Longs Data ---
    if not os.path.exists(LONGS_DATA_PATH):
        print(f"ERROR: Longs data file not found at {LONGS_DATA_PATH}. Exiting.")
        return
    print(f"Loading longs data from: {os.path.basename(LONGS_DATA_PATH)}")
    longs_df = pd.read_parquet(LONGS_DATA_PATH)
    longs_df['trade_direction'] = 'LONG'

    # --- Load Shorts Data ---
    if not os.path.exists(SHORTS_DATA_PATH):
        print(f"ERROR: Shorts data file not found at {SHORTS_DATA_PATH}. Exiting.")
        return
    print(f"Loading shorts data from: {os.path.basename(SHORTS_DATA_PATH)}")
    shorts_df = pd.read_parquet(SHORTS_DATA_PATH)
    shorts_df['trade_direction'] = 'SHORT'

    # --- Combine and Save ---
    print("\nCombining long and short data into a unified DataFrame...")
    unified_df = pd.concat([longs_df, shorts_df], ignore_index=True)
    
    # Sort for good practice, although the simulator will re-sort
    unified_df.sort_values(by=['datetime', 'symbol'], inplace=True)

    output_path = os.path.join(DATA_DIRECTORY, OUTPUT_FILENAME)
    
    try:
        unified_df.to_parquet(output_path)
        print(f"\n--- Success! ---")
        print(f"Successfully created unified data file: '{OUTPUT_FILENAME}'")
        print("The v2.0 unified simulator is now ready to use this file.")
    except Exception as e:
        print(f"\nERROR: An error occurred while saving the file: {e}")

if __name__ == "__main__":
    main()
