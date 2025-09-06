# fix_corrupted_data.py (Enhanced for Portability and Advanced Cleaning)
#
# Description:
# This script scans a directory for corrupted CSV files and cleans them.
# It is now portable and can be run from a subdirectory (e.g., 'utilities').
# It fixes two known corruption issues:
#   1. Files with an extra 'datetime.1' index column from an old bug.
#   2. Files with rows that have a missing datetime value.
#
# INSTRUCTIONS:
# 1. Place this script in your project's 'utilities' directory.
# 2. Run it once from your terminal: python utilities/fix_corrupted_data.py
# 3. This will permanently fix all corrupted data files.

import os
import sys
import pandas as pd

# --- Project Root Configuration for Portability ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Fallback for interactive environments
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, PROJECT_ROOT)


# --- CONFIGURATION (with portable paths) ---
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "universal_historical_data")

def clean_csv_file(filepath):
    """
    Reads a CSV file, checks for multiple types of corruption, and fixes them in place.
    """
    was_modified = False
    df = None

    try:
        # First, check for the corrupted header with 'datetime.1'
        header = pd.read_csv(filepath, nrows=0).columns.tolist()
        if 'datetime' in header and 'datetime.1' in header:
            tqdm.write(f"  - Fixing header in: {os.path.basename(filepath)}")
            df = pd.read_csv(filepath, usecols=['datetime.1', 'open', 'high', 'low', 'close', 'volume'])
            df.rename(columns={'datetime.1': 'datetime'}, inplace=True)
            was_modified = True
        else:
            # If header is fine, read the file normally
            df = pd.read_csv(filepath)

        # Next, check for and remove rows with missing datetime values
        initial_rows = len(df)
        df.dropna(subset=['datetime'], inplace=True)
        final_rows = len(df)

        if initial_rows != final_rows:
            rows_removed = initial_rows - final_rows
            tqdm.write(f"  - Removed {rows_removed} bad rows from: {os.path.basename(filepath)}")
            was_modified = True

        # If any fixes were applied, overwrite the original file
        if was_modified:
            df.to_csv(filepath, index=False)
            return True
            
        return False

    except pd.errors.EmptyDataError:
        tqdm.write(f"  - WARNING: Skipping empty file: {os.path.basename(filepath)}")
        return False
    except Exception as e:
        tqdm.write(f"  - ERROR processing {os.path.basename(filepath)}: {e}")
        return False

def main():
    """
    Main function to find and clean all CSV files in the target directory.
    """
    print(f"--- Starting Data Cleaning Process ---")
    print(f"Target Directory: {DATA_DIRECTORY}\n")
    
    if not os.path.isdir(DATA_DIRECTORY):
        print(f"Error: Directory not found at '{DATA_DIRECTORY}'. Please check the path.")
        return

    all_files = []
    for root, _, files in os.walk(DATA_DIRECTORY):
        for file in files:
            if file.endswith('.csv'):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print("No CSV files found to clean.")
        return

    # Use tqdm for a progress bar
    from tqdm import tqdm
    cleaned_count = 0
    for filepath in tqdm(all_files, desc="Cleaning files"):
        if clean_csv_file(filepath):
            cleaned_count += 1
    
    print("\n--- Data Cleaning Complete! ---")
    print(f"Scanned {len(all_files)} CSV files.")
    print(f"Fixed {cleaned_count} corrupted files.")

if __name__ == "__main__":
    # Add tqdm to dependencies check
    try:
        from tqdm import tqdm
    except ImportError:
        print("FATAL ERROR: tqdm library NOT FOUND. Please install it using: pip install tqdm")
        sys.exit(1)
    main()

