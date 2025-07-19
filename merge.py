import os
import pandas as pd
import glob

# --- Configuration ---
input_folder = 'daily_with_indicators'  # <-- Set your folder path here
file_pattern = '*_daily_with_indicators.csv'
output_filename = 'merged_and_sorted_stocks_optimized.csv'
date_column_name = 'datetime'
stock_column_name = 'stock_ticker'

# 1. Create the full search path
search_path = os.path.join(input_folder, file_pattern)
print(f"Searching for files in: {search_path}")

# 2. Get a list of all CSV files matching the pattern
csv_files = glob.glob(search_path)

if not csv_files:
    print(f"Error: No CSV files found in the folder: '{input_folder}'")
    print("Please check that the `input_folder` path is correct and that it contains matching CSV files.")
else:
    print(f"Found {len(csv_files)} files to merge: {csv_files}")

    # 3. Read, process, and collect each CSV into a list of DataFrames
    list_of_dfs = []
    for file in csv_files:
        base_name = os.path.basename(file)
        stock_ticker = base_name.split('_')[0]
        df = pd.read_csv(file)
        df[stock_column_name] = stock_ticker
        list_of_dfs.append(df)

    # 4. Concatenate all DataFrames into a single one
    merged_df = pd.concat(list_of_dfs, ignore_index=True)

    # 5. (SKIP: Do not convert the date column to datetime objects for proper sorting)

    # 6. Sort the DataFrame by the date column in ascending order
    sorted_df = merged_df.sort_values(by=date_column_name)

    # --- Filter to last 3 months only ---
    # Assumes date_column_name is in 'YYYY-MM-DD' or similar sortable string format
    last_date = sorted_df[date_column_name].max()
    three_months_ago = pd.to_datetime(last_date) - pd.DateOffset(months=3)
    sorted_df = sorted_df[pd.to_datetime(sorted_df[date_column_name]) >= three_months_ago]

    # --- Column Filtering Logic ---
    def filter_columns(df):
        # Columns to always keep (adjust as needed)
        keep_cols = [stock_column_name, date_column_name, 'open', 'high', 'low', 'close', 'volume']
        # Add EMA 20 and EMA 30 columns (case-insensitive match)
        for col in df.columns:
            col_lower = col.lower()
            if 'ema_20' in col_lower or 'ema_30' in col_lower:
                keep_cols.append(col)
        # Remove duplicates in keep_cols
        keep_cols = list(dict.fromkeys(keep_cols))
        # Filter out MACD and Bollinger Band columns
        filtered = df[[col for col in keep_cols if col in df.columns]]
        return filtered

    sorted_df = filter_columns(sorted_df)

    # --- Further Optimizations ---
    # Downcast numeric columns
    for col in sorted_df.select_dtypes(include=['float64', 'int64']).columns:
        sorted_df[col] = pd.to_numeric(sorted_df[col], downcast='float')
    # Convert string columns to category
    for col in sorted_df.select_dtypes(include=['object']).columns:
        sorted_df[col] = sorted_df[col].astype('category')

    # Save as regular CSV (no .gz extension)
    sorted_df.to_csv(output_filename, index=False)
    print(f"Optimized file saved as {output_filename}. Size: {round(os.path.getsize(output_filename) / (1024*1024), 2)} MB")