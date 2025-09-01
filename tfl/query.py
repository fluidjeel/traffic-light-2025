import pandas as pd
import os
import warnings

# --- SUPPRESS FUTUREWARNING ---
# This script uses standard pandas features, but this keeps the output clean.
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# This script should be run from the root of your project (e.g., D:\algo-2025)
ROOT_DIR = os.getcwd()
# This must point to the final output file from your data pipeline script
DATA_FILE_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_longs_data_with_signals_v2.parquet")

# ==============================================================================
# --- QUERY SCRIPT ---
# ==============================================================================

def find_and_print_samples(data_df):
    """
    Queries the DataFrame to find and print two samples of actual generated
    'fast entry' signals, grouped by the number of red candles in their pattern.
    """
    print("--- TFL Entry Signal Validation Tool ---")

    # ENHANCEMENT: Instead of looking for setups, we now look for actual,
    # confirmed 'fast entry' signals that the backtester would trade.
    # We also fill any potential missing boolean values (NaN) with False.
    valid_signals = data_df[data_df['is_fast_entry'].fillna(False)].copy()

    if valid_signals.empty:
        print("\nNo valid 'fast entry' signals were found in the data file.")
        return

    # BUG FIX: To correctly find the red candle count for each signal's setup
    # candle (T-1), we must perform the shift operation on a per-symbol basis.
    # This is done by grouping by 'symbol' before applying the shift. This
    # resolves the "cannot reindex on an axis with duplicate labels" error.
    data_df['setup_red_candle_count'] = data_df.groupby('symbol')['pattern_red_candle_count'].shift(1)
    
    # Now, we can safely select the valid signals and access this new column.
    valid_signals = data_df[data_df['is_fast_entry'].fillna(False)].copy()

    # Loop from 1 up to a reasonable maximum (e.g., 9) to find samples for each pattern length
    for i in range(1, 10):
        print(f"\n\n--- Querying for 'fast entry' signals from patterns with exactly {i} red candle(s) ---")
        
        # Filter the signals DataFrame for entries that came from a pattern of the current length
        pattern_subset = valid_signals[valid_signals['setup_red_candle_count'] == i]
        
        if pattern_subset.empty:
            print(f"No entry signals found for patterns with {i} red candle(s).")
            continue
            
        # Try to get 2 random samples. If fewer than 2 exist, get what's available.
        num_samples = min(2, len(pattern_subset))
        
        # Using a fixed random_state ensures reproducibility
        samples = pattern_subset.sample(n=num_samples, random_state=42)
        
        print(f"Found {len(pattern_subset)} total signals. Displaying {num_samples} sample(s):")
        # Display relevant columns for validation
        print(samples[['symbol', 'daily_rsi', 'setup_red_candle_count', 'fast_entry_price']])

def main():
    """
    Main function to load data and run the query.
    """
    if not os.path.exists(DATA_FILE_PATH):
        print(f"ERROR: Data file not found at the specified path.")
        print(f"Please check that this file exists: {DATA_FILE_PATH}")
        return
        
    try:
        # Load the entire Parquet file into memory
        master_df = pd.read_parquet(DATA_FILE_PATH)
        # Ensure the datetime column is treated as a timezone-aware index for sorting
        if 'datetime' in master_df.columns:
            master_df['datetime'] = pd.to_datetime(master_df['datetime'])
            # Sort values to ensure correct shifting within groups
            master_df.sort_values(by=['symbol', 'datetime'], inplace=True)
            master_df.set_index('datetime', inplace=True)
            
        find_and_print_samples(master_df)

    except Exception as e:
        print(f"\nAn error occurred while loading or processing the file: {e}")


if __name__ == "__main__":
    main()

