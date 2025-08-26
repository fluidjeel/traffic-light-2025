# verify_single_parquet.py
#
# Description:
# A utility script to verify the integrity and cleanliness of a single
# Parquet file, specifically checking the datetime index.
#
# INSTRUCTIONS:
# 1. Place this script in your project's root directory.
# 2. Run this script to verify a specific Parquet file.

import pandas as pd
import os

# --- CONFIGURATION ---
# The project root directory
ROOT_DIR = "D:\\algo-2025"

# The path to the specific Parquet file to verify
FILE_TO_VERIFY = os.path.join(ROOT_DIR, "data", "universal_processed", "15min", "ABB_15min_with_indicators.parquet")

def verify_file():
    """
    Performs a series of integrity checks on a single Parquet file.
    """
    print(f"--- Starting Verification for {os.path.basename(FILE_TO_VERIFY)} ---")

    if not os.path.exists(FILE_TO_VERIFY):
        print(f"❌ ERROR: File not found at: {FILE_TO_VERIFY}")
        return False
    
    file_size_mb = os.path.getsize(FILE_TO_VERIFY) / (1024 * 1024)
    print(f"✅ File found. Size: {file_size_mb:.2f} MB")

    try:
        df = pd.read_parquet(FILE_TO_VERIFY)
        print(f"✅ File loaded successfully. Total rows: {len(df)}")
        
        if df.empty:
            print("❌ ERROR: The DataFrame is empty.")
            return False

        # Check if the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            print("❌ ERROR: The DataFrame index is not a DatetimeIndex.")
            return False
        
        # Check for NaT values in the index
        if pd.isna(df.index).any():
            print("❌ ERROR: The DataFrame index contains NaT (Not a Time) values.")
            return False

        # Check for chronological order
        if not df.index.is_monotonic_increasing:
            print("❌ ERROR: The DataFrame index is not in chronological order.")
            return False

        print("\n✅ Verification Complete! File is clean and ready for use.")
        return True

    except Exception as e:
        print(f"❌ An unexpected error occurred during verification: {e}")
        return False

if __name__ == "__main__":
    verify_file()
