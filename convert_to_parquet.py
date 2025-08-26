# convert_to_parquet.py
#
# Description:
# This script scans the data directory for all CSV files with indicators,
# reads them, and saves them in the much faster Parquet format. This
# helps to significantly speed up the backtester's data loading time.
#
# INSTRUCTIONS:
# 1. Make sure you have pyarrow installed: pip install pyarrow
# 2. Place this script in your project's root directory (e.g., D:\algo-2025).
# 3. Run it once from your terminal: python convert_to_parquet.py

import os
import pandas as pd
from multiprocessing import Pool, cpu_count
import sys

# --- CONFIGURATION ---
# The directory where your processed CSV files are located.
# The script will scan all subfolders within this directory.
SOURCE_DIRECTORY = os.path.join("data", "universal_processed")

def convert_file_to_parquet(csv_filepath):
    """
    Reads a single CSV file and saves it as a Parquet file.
    """
    try:
        # Construct the new filename
        parquet_filepath = csv_filepath.replace('.csv', '.parquet')
        
        # Read the CSV, ensuring the datetime column is parsed correctly
        df = pd.read_csv(csv_filepath, index_col='datetime', parse_dates=True)
        
        if df.empty:
            print(f"  - Skipping empty file: {os.path.basename(csv_filepath)}")
            return
            
        # Save to Parquet format
        df.to_parquet(parquet_filepath)
        print(f"  - Converted: {os.path.basename(parquet_filepath)}")
        
    except Exception as e:
        print(f"  - Error processing {os.path.basename(csv_filepath)}: {e}")

def main():
    """
    Finds all CSV files and converts them in parallel.
    """
    print("--- Starting CSV to Parquet Conversion ---")
    print(f"Scanning directory: {SOURCE_DIRECTORY}\n")
    
    if not os.path.isdir(SOURCE_DIRECTORY):
        print(f"Error: Directory not found: '{SOURCE_DIRECTORY}'")
        return
        
    # Find all CSV files that need conversion
    files_to_convert = []
    for root, _, files in os.walk(SOURCE_DIRECTORY):
        for file in files:
            if file.endswith("_with_indicators.csv"):
                files_to_convert.append(os.path.join(root, file))
    
    if not files_to_convert:
        print("No CSV files with indicators found to convert.")
        return
        
    print(f"Found {len(files_to_convert)} files to convert...")
    
    # Use multiprocessing to speed up the conversion
    num_processes = max(1, cpu_count() - 1)
    try:
        with Pool(processes=num_processes) as pool:
            pool.map(convert_file_to_parquet, files_to_convert)
        
        print("\n--- Conversion Complete! ---")

    except KeyboardInterrupt:
        print("\n--- Process interrupted by user (Ctrl+C). Terminating workers and exiting. ---")
        # The 'with' statement automatically handles the termination of the pool.
        sys.exit(1)

if __name__ == "__main__":
    main()
