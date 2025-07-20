import pandas as pd
import os

# --- CONFIGURATION ---
# The source file containing stock symbols and their industry/sector
SOURCE_CSV = 'nifty200-industry.csv'
# The clean, two-column output file that the backtester will use
OUTPUT_CSV = 'stock_to_sector.csv'

# The names of the columns in your source CSV
# IMPORTANT: Change these if your column names are different
SYMBOL_COLUMN = 'Symbol'
SECTOR_COLUMN = 'Industry'

def main():
    """
    Reads the Nifty 200 list, extracts the symbol and sector,
    and creates a clean mapping file for the backtester.
    """
    try:
        df = pd.read_csv(SOURCE_CSV)
        print(f"Successfully loaded {SOURCE_CSV}")
    except FileNotFoundError:
        print(f"Error: Source file not found at '{SOURCE_CSV}'. Please ensure it is in the correct directory.")
        return

    # Check if the required columns exist
    if SYMBOL_COLUMN not in df.columns or SECTOR_COLUMN not in df.columns:
        print(f"Error: The required columns '{SYMBOL_COLUMN}' and/or '{SECTOR_COLUMN}' were not found in the CSV.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    # Create the mapping DataFrame
    mapping_df = df[[SYMBOL_COLUMN, SECTOR_COLUMN]].copy()
    mapping_df.rename(columns={SYMBOL_COLUMN: 'symbol', SECTOR_COLUMN: 'sector'}, inplace=True)

    # Clean up sector names to create valid Fyers API symbols (e.g., "NIFTY AUTO")
    # This replaces spaces and special characters with underscores and makes it uppercase
    mapping_df['sector_symbol'] = 'NIFTY ' + mapping_df['sector'].str.upper().str.replace(' ', '_').str.replace('[^A-Z0-9_]', '', regex=True)
    
    # Save the clean mapping file
    mapping_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSuccessfully created the stock-to-sector mapping file: {OUTPUT_CSV}")
    print(f"Found {len(mapping_df['sector_symbol'].unique())} unique sectors.")
    print("\nNext steps:")
    print("1. Review the 'sector_symbol' column in 'stock_to_sector.csv' to ensure the names are correct.")
    print("2. Use the 'download_sector_data.py' script to fetch data for these sector indices.")


if __name__ == "__main__":
    main()
