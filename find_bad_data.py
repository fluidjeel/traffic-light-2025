# fix_corrupted_csv_data.py
#
# Description:
# This script scans the raw data directory, loads each CSV,
# cleans any malformed or missing datetime entries, and saves the
# corrected file. This is a critical step to fix the "NaT" errors
# at their source.
#
# INSTRUCTIONS:
# 1. Place this script in your project's root directory.
# 2. Run this script ONCE to clean all your raw data files.
# 3. After this, you can re-run your entire pipeline from the beginning.

import pandas as pd
import os
import glob
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
# This is the directory where your raw CSV files are located.
ROOT_DIR = "D:\\algo-2025"
DATA_DIRECTORY_15MIN = os.path.join(ROOT_DIR, "data", "universal_historical_data")

def clean_single_csv_file(file_path):
    """
    Loads a single CSV, cleans the 'datetime' column, and saves the corrected file.
    """
    file_name = os.path.basename(file_path)
    print(f"  - Cleaning: {file_name}")

    try:
        # Read the CSV with flexible parsing
        df = pd.read_csv(file_path)
        
        # Force the 'datetime' column to a proper datetime format, coercing errors to NaT
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        
        # Drop any rows where the datetime parsing failed
        df.dropna(subset=['datetime'], inplace=True)
        
        # Set the cleaned datetime column as the index
        df.set_index('datetime', inplace=True)

        # Check if the cleaned DataFrame is still valid
        if df.empty:
            print(f"    -> Warning: File for {file_name} is now empty after cleaning. Skipping save.")
            return
            
        # Sort the index to ensure chronological order
        df.sort_index(inplace=True)

        # Save the corrected file, overwriting the original
        df.to_csv(file_path, index=True)
        print(f"    -> Success: {file_name} cleaned and saved.")
        
    except Exception as e:
        print(f"    -> Error processing {file_name}: {e}. Skipping file.")

def main():
    """Main function to iterate and clean all CSV files in parallel."""
    print("--- Starting Bulk CSV Data Cleaning ---")
    
    if not os.path.isdir(DATA_DIRECTORY_15MIN):
        print(f"Error: Directory not found: '{DATA_DIRECTORY_15MIN}'")
        return

    files_to_clean = glob.glob(os.path.join(DATA_DIRECTORY_15MIN, "*_15min.csv"))
    
    if not files_to_clean:
        print("No CSV files found to clean.")
        return

    print(f"Found {len(files_to_clean)} files to clean...")
    
    num_processes = max(1, cpu_count() - 1)
    try:
        with Pool(processes=num_processes) as pool:
            pool.map(clean_single_csv_file, files_to_clean)
        
        print("\n--- CSV Data Cleaning Complete! ---")
        print("\nYou can now proceed with re-running your entire data pipeline from Step 1.")

    except KeyboardInterrupt:
        print("\n--- Process interrupted by user (Ctrl+C). Terminating workers and exiting. ---")
        sys.exit(1)

if __name__ == "__main__":
    main()
