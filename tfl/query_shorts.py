import pandas as pd
import os
import warnings

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# This script should be run from the root of your project (e.g., D:\algo-2025)
ROOT_DIR = os.getcwd()
# This must point to the final output file from your SHORTS data pipeline
DATA_FILE_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_shorts_data_with_signals_and_atr.parquet")

# ==============================================================================
# --- QUERY SCRIPT ---
# ==============================================================================

def find_and_print_samples(data_df):
    """
    Queries the DataFrame to find and print two samples of actual generated
    'fast entry' signals, grouped by the number of GREEN candles in their pattern.
    """
    print("--- TFL Short Entry Signal Validation Tool ---")

    valid_signals = data_df[data_df['is_fast_entry'].fillna(False)].copy()

    if valid_signals.empty:
        print("\nNo valid 'fast entry' signals were found in the data file.")
        return

    # For shorts, we look for the 'pattern_green_candle_count' on the setup candle (T-1)
    data_df['setup_green_candle_count'] = data_df.groupby('symbol')['pattern_green_candle_count'].shift(1)
    
    # Re-select the valid signals to include this new column
    valid_signals = data_df[data_df['is_fast_entry'].fillna(False)].copy()

    # Loop from 1 to 9 to find samples for each pattern length
    for i in range(1, 10):
        print(f"\n\n--- Querying for 'fast entry' signals from patterns with exactly {i} green candle(s) ---")
        
        # Filter for entries that came from a pattern of the current length
        pattern_subset = valid_signals[valid_signals['setup_green_candle_count'] == i]
        
        if pattern_subset.empty:
            print(f"No entry signals found for patterns with {i} green candle(s).")
            continue
            
        num_samples = min(2, len(pattern_subset))
        samples = pattern_subset.sample(n=num_samples, random_state=42)
        
        print(f"Found {len(pattern_subset)} total signals. Displaying {num_samples} sample(s):")
        # Display relevant columns for validation
        print(samples[['symbol', 'daily_rsi', 'setup_green_candle_count', 'fast_entry_price']])

def main():
    """
    Main function to load data and run the query.
    """
    if not os.path.exists(DATA_FILE_PATH):
        print(f"ERROR: Data file not found at the specified path.")
        print(f"Please check that this file exists: {DATA_FILE_PATH}")
        return
        
    try:
        master_df = pd.read_parquet(DATA_FILE_PATH)
        if 'datetime' in master_df.columns:
            master_df['datetime'] = pd.to_datetime(master_df['datetime'])
            master_df.sort_values(by=['symbol', 'datetime'], inplace=True)
            master_df.set_index('datetime', inplace=True)
            
        find_and_print_samples(master_df)

    except Exception as e:
        print(f"\nAn error occurred while loading or processing the file: {e}")


if __name__ == "__main__":
    main()
