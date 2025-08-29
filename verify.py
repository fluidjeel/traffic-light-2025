import os
import pandas as pd
from pytz import timezone

# --- CONFIGURATION (Match your simulator settings) ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_longs_data_with_signals.parquet")

# --- PARAMETERS TO INVESTIGATE ---
SYMBOL = 'TECHM'
DATE_TO_CHECK = '2018-02-05'

def check_data_issue():
    """Queries the Parquet file to check for data inconsistencies."""
    print(f"--- Investigating data for {SYMBOL} on {DATE_TO_CHECK} ---")
    try:
        # Load the partitioned Parquet file
        df = pd.read_parquet(DATA_PATH, filters=[('symbol', '==', SYMBOL)])
        
        # Ensure the index is a datetime object
        if not isinstance(df.index, pd.DatetimeIndex):
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
        # Convert the index to the correct timezone
        df.index = df.index.tz_convert(timezone('Asia/Kolkata'))
        
        # Filter for the specific date
        df_filtered = df.loc[DATE_TO_CHECK].copy()
        
        if df_filtered.empty:
            print("No data found for the specified symbol and date.")
            return

        # Select and print relevant columns
        cols_to_print = [
            'open', 'high', 'low', 'close',
            'is_entry_signal', 'pattern_high_trigger', 'pattern_low_trigger'
        ]
        
        print("\nCandle Data and Signal Information:")
        print(df_filtered[cols_to_print].to_string())

        # Check for the specific SL value from the log
        target_sl_value = 620.0
        if target_sl_value in df_filtered['pattern_low_trigger'].values:
            print(f"\n✅ The target stop-loss value of {target_sl_value} was found as a pattern low.")
        else:
            print(f"\n❌ The target stop-loss value of {target_sl_value} was NOT found as a pattern low. This confirms a data issue.")

    except FileNotFoundError:
        print(f"ERROR: Parquet file not found at {DATA_PATH}. Please ensure it has been created.")
    except KeyError:
        print(f"ERROR: Symbol '{SYMBOL}' not found in the Parquet file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    check_data_issue()
