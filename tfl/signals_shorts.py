import pandas as pd
import os
import sys

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# This script should be run from the root of your project (e.g., D:\algo-2025)
ROOT_DIR = os.getcwd()
DATA_DIRECTORY = os.path.join(ROOT_DIR, "data", "strategy_specific_data")

# List of SHORTS Parquet files you want to inspect
FILES_TO_PROCESS = [
    "tfl_shorts_data_with_signals_v2.parquet",
    "tfl_shorts_data_with_signals_and_atr.parquet"
]

# The column that contains the boolean flag for the entry signal
# Based on the symmetrical pipeline, this is 'is_fast_entry' for shorts as well.
SIGNAL_COLUMN = "is_fast_entry"

# ==============================================================================
# --- SCRIPT LOGIC ---
# ==============================================================================

def main():
    """
    Loads each specified SHORTS Parquet file, finds all rows with a true entry signal,
    and saves them to a corresponding CSV file for easy inspection.
    """
    print("--- TFL Short Signal Extraction Utility ---")

    for filename in FILES_TO_PROCESS:
        input_path = os.path.join(DATA_DIRECTORY, filename)
        output_filename = filename.replace('.parquet', '_signals_only.csv')
        output_path = os.path.join(DATA_DIRECTORY, output_filename)

        print(f"\nProcessing file: {filename}...")

        if not os.path.exists(input_path):
            print(f"  - ERROR: File not found at {input_path}. Skipping.")
            continue

        try:
            # Load the data file
            df = pd.read_parquet(input_path)
            print(f"  - Successfully loaded {len(df)} rows.")

            # Find all rows where the signal column is True
            signals_df = df[df[SIGNAL_COLUMN].fillna(False)].copy()

            if signals_df.empty:
                print("  - No active signals found in this file.")
                continue

            print(f"  - Found {len(signals_df)} rows with active signals.")
            
            # Save the results to a new CSV file
            signals_df.to_csv(output_path, index=False)
            print(f"  - Successfully saved signals to: {output_filename}")

        except Exception as e:
            print(f"  - ERROR: An unexpected error occurred while processing the file: {e}")

    print("\n--- Extraction Complete ---")

if __name__ == "__main__":
    main()
