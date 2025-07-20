import pandas as pd
import os
import sys

# --- CONFIGURATION ---
# Directory where raw daily data is stored (output from your scraper)
SOURCE_DATA_DIR = 'historical_data'
# Base directory where the processed files with indicators will be saved
BASE_DESTINATION_DIR = 'data/processed' # A base folder to hold all timeframes
# The CSV file containing the list of symbols to process
NIFTY_LIST_CSV = 'nifty200.csv'

def calculate_atr(df, period=14):
    """Calculates the Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def main():
    """
    Main function to iterate through stock data files, resample to multiple
    timeframes, calculate indicators, and save the processed files.
    """
    try:
        symbols_df = pd.read_csv(NIFTY_LIST_CSV)
        symbols = symbols_df.iloc[:, 0].tolist()
        # Also process the NIFTY200 index for the regime filter
        symbols.append('NIFTY200_INDEX') 
    except FileNotFoundError:
        print(f"Error: {NIFTY_LIST_CSV} not found.")
        sys.exit()

    # Define the timeframes and their resampling rules
    timeframes = {
        'daily': 'D',
        '2day': '2D',
        'weekly': 'W-MON', # Weekly, starting on Monday
        'monthly': 'MS'     # Monthly, starting on the first day of the month
    }

    # Aggregation rules for resampling OHLCV data
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    print(f"Found {len(symbols)} symbols to process across {len(timeframes)} timeframes...")

    for i, symbol in enumerate(symbols):
        source_path = None # Initialize source path
        
        # Flexibly find the source data file
        if symbol.endswith('_INDEX'):
             source_filename = f"{symbol}_daily.csv"
             source_path = os.path.join(SOURCE_DATA_DIR, source_filename)
        else:
             source_filename_eq = f"{symbol}_EQ_daily.csv"
             source_filename_plain = f"{symbol}_daily.csv"
             path_eq = os.path.join(SOURCE_DATA_DIR, source_filename_eq)
             path_plain = os.path.join(SOURCE_DATA_DIR, source_filename_plain)

             if os.path.exists(path_eq):
                 source_path = path_eq
             elif os.path.exists(path_plain):
                 source_path = path_plain
             else:
                 print(f"({i+1}/{len(symbols)}) Warning: Source file not found for {symbol}. Skipping.")
                 continue
        
        if not os.path.exists(source_path):
            continue

        try:
            # Load the base daily data once
            df_daily = pd.read_csv(source_path, index_col=0, parse_dates=True)
            df_daily.columns = [col.lower() for col in df_daily.columns]

            # Process for each defined timeframe
            for tf_name, tf_rule in timeframes.items():
                
                # Create the specific destination directory for the timeframe
                dest_dir = os.path.join(BASE_DESTINATION_DIR, tf_name)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                df = None
                if tf_name == 'daily':
                    df = df_daily.copy()
                else:
                    # Resample the daily data to the target timeframe
                    df = df_daily.resample(tf_rule).agg(agg_rules)
                
                df.dropna(inplace=True) # Remove rows with no data after resampling
                if df.empty:
                    continue

                # --- CALCULATE INDICATORS ---
                # 1. Exponential Moving Averages (EMAs)
                df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
                df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
                # ema_100 has been removed as it is not used by the backtester

                # 2. Average True Range (ATR) for Volatility Filter
                df['atr_14'] = calculate_atr(df, period=14)
                df['atr_ma_30'] = df['atr_14'].rolling(window=30).mean()

                # --- SAVE PROCESSED FILE ---
                dest_filename = f"{symbol.replace('_INDEX', '')}_daily_with_indicators.csv" if not symbol.endswith('_INDEX') else f"{symbol}_daily_with_indicators.csv"
                dest_path = os.path.join(dest_dir, dest_filename)
                df.to_csv(dest_path)
            
            print(f"({i+1}/{len(symbols)}) Processed {symbol} for all timeframes.")

        except Exception as e:
            print(f"({i+1}/{len(symbols)}) Error processing {symbol}: {e}")

    print("\nIndicator calculation complete for all timeframes.")

if __name__ == "__main__":
    main()
