import os
import pandas as pd

def find_missing_stock_data(nifty_csv_path, data_directory, interval="daily", cleanup=False):
    """
    Finds which stocks from a CSV file are missing their corresponding data files
    for a specified interval, with optional cleanup of source CSV.

    Args:
        nifty_csv_path (str): Path to the CSV file containing stock symbols.
        data_directory (str): Directory where scraped data is stored.
        interval (str): Data interval to check (e.g., "daily", "15min").
        cleanup (bool): If True, removes missing symbols from source CSV.
    """
    try:
        # Read the list of stocks from the CSV file
        nifty_stocks_df = pd.read_csv(nifty_csv_path)
        symbols = nifty_stocks_df['Symbol'].tolist()

        missing_stocks = []
        found_stocks = []

        # Iterate through each symbol and check for its corresponding data file
        for symbol in symbols:
            filename = f"{symbol}_{interval}.csv"
            file_path = os.path.join(data_directory, filename)

            if os.path.exists(file_path):
                found_stocks.append(symbol)
            else:
                missing_stocks.append(symbol)

        print(f"--- Stock Data Verification Report for {interval} data ---")
        print(f"Total stocks in {nifty_csv_path}: {len(symbols)}\n")

        if missing_stocks:
            print("The following stocks are missing their data files:")
            for stock in missing_stocks:
                print(f"- {stock}")
            print(f"\nTotal missing: {len(missing_stocks)}")
            
            if cleanup:
                # Create backup of original file
                backup_path = nifty_csv_path.replace('.csv', '_backup.csv')
                nifty_stocks_df.to_csv(backup_path, index=False)
                print(f"\nCreated backup at: {backup_path}")
                
                # Filter out missing symbols
                updated_df = nifty_stocks_df[~nifty_stocks_df['Symbol'].isin(missing_stocks)]
                updated_df.to_csv(nifty_csv_path, index=False)
                print(f"✅ Removed {len(missing_stocks)} missing symbols from {nifty_csv_path}")
        else:
            print(f"✅ All {interval} data files were found.")

        print(f"\nTotal found: {len(found_stocks)}")

    except FileNotFoundError:
        print(f"Error: The file '{nifty_csv_path}' or directory '{data_directory}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Example Usage ---
NIFTY_CSV_FILE = 'nifty500.csv'
DATA_DIR = os.path.join('data', 'universal_historical_data')

# Check for 15-minute data with cleanup enabled
find_missing_stock_data(
    nifty_csv_path=NIFTY_CSV_FILE,
    data_directory=DATA_DIR,
    interval="15min",
    cleanup=True  # Set to False for dry-run
)