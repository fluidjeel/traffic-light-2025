# trade_auditor.py
#
# Description:
# This script audits the results of a trading strategy backtest to ensure its logical integrity.
# It automatically finds the latest backtest results for a given strategy and cross-references
# each trade from the trade_log.csv against the original 15-minute price data.
# The audit verifies that entries, exits, and PnL calculations were executed correctly.
#
# Author: Gemini
# Version: 1.4.0 (Corrected premature exit logic)

import pandas as pd
from pathlib import Path
import numpy as np

# --- Configuration ---
# The script assumes it's located in /project_root/tfl/
# We navigate up one level to establish the project's root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the strategy to be audited. This should match the folder name in /backtest_logs/.
STRATEGY_NAME = "TrafficLight-Manny-SHORTS_ONLY"

# Define key directory paths relative to the project root
BACKTEST_LOGS_DIR = PROJECT_ROOT / "backtest_logs"
PRICE_DATA_DIR = PROJECT_ROOT / "data" / "universal_processed" / "15min"

# Define a small tolerance for floating-point comparisons of PnL.
PNL_TOLERANCE = 0.01
# Set to True if the strategy being audited is SHORTS ONLY.
# This determines the logic for SL/TP hits and PnL calculation.
IS_SHORT_STRATEGY = True

# --- Helper Functions ---

def find_latest_log_path(base_dir: Path, strategy_name: str) -> Path:
    """
    Finds the most recent backtest log file for a given strategy by sorting
    the timestamped result folders.

    Args:
        base_dir: The root directory containing all backtest logs.
        strategy_name: The name of the strategy to find logs for.

    Returns:
        A Path object pointing to the latest trade_log.csv file.

    Raises:
        FileNotFoundError: If the strategy directory, backtest folders, or log file cannot be found.
    """
    strategy_dir = base_dir / strategy_name
    if not strategy_dir.exists():
        raise FileNotFoundError(f"Strategy directory not found: {strategy_dir}")

    # Identify all subdirectories which represent individual backtest runs
    backtest_folders = [d for d in strategy_dir.iterdir() if d.is_dir()]
    if not backtest_folders:
        raise FileNotFoundError(f"No backtest folders found in {strategy_dir}")

    # Sort folders by name to find the most recent one (YYYYMMDD_HHMMSS format sorts chronologically)
    latest_folder = sorted(backtest_folders, reverse=True)[0]
    log_file = latest_folder / "trade_log.csv"

    if not log_file.exists():
        raise FileNotFoundError(f"trade_log.csv not found in {latest_folder}")

    print(f"✅ Found latest log file: {log_file}")
    return log_file

# --- Main Audit Logic ---

def audit_strategy(strategy_name: str):
    """
    Executes the full audit process for a given trading strategy.

    Args:
        strategy_name: The name of the strategy to audit.
    """
    try:
        trade_log_path = find_latest_log_path(BACKTEST_LOGS_DIR, strategy_name)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Load the trade log and prepare the data
    trade_log_df = pd.read_csv(trade_log_path)
    
    # --- FIX for Timezone Error (v1.3.0) ---
    # Convert time columns to datetime objects.
    trade_log_df['entry_time'] = pd.to_datetime(trade_log_df['entry_time'], errors='coerce')
    trade_log_df['exit_time'] = pd.to_datetime(trade_log_df['exit_time'], errors='coerce')

    # Immediately make the datetime columns timezone-naive to prevent comparison errors.
    # This handles cases where the CSV might have timezone info.
    if pd.api.types.is_datetime64_any_dtype(trade_log_df['entry_time']):
        trade_log_df['entry_time'] = trade_log_df['entry_time'].dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(trade_log_df['exit_time']):
        trade_log_df['exit_time'] = trade_log_df['exit_time'].dt.tz_localize(None)

    trade_log_df.dropna(subset=['entry_time', 'exit_time'], inplace=True)

    discrepancies = []
    total_trades = len(trade_log_df)
    
    # Cache for price data to avoid reloading the same file multiple times
    price_data_cache = {}

    print(f"\n🚀 Starting audit for {total_trades} trades...")

    for _, trade in trade_log_df.iterrows():
        symbol = trade['symbol']
        
        # --- 1. Load Price Data ---
        # Load corresponding price data if not already in the cache
        if symbol not in price_data_cache:
            price_file_path = PRICE_DATA_DIR / f"{symbol}_15min_with_indicators.parquet"
            if not price_file_path.exists():
                print(f"⚠️ Warning: Price data not found for {symbol}. Skipping trades for this symbol.")
                discrepancies.append({
                    "trade": trade.to_dict(),
                    "reason": "Missing Price Data",
                    "details": f"Parquet file not found at {price_file_path}"
                })
                continue
            
            price_df = pd.read_parquet(price_file_path)
            
            if not isinstance(price_df.index, pd.DatetimeIndex):
                print(f"⚠️ Warning: Index for {symbol} is not a DatetimeIndex. Attempting to convert.")
                price_df.index = pd.to_datetime(price_df.index)

            # --- FIX for Timezone Error (v1.3.0) ---
            # Ensure the price data index is also timezone-naive.
            if price_df.index.tz is not None:
                price_df.index = price_df.index.tz_localize(None)

            price_data_cache[symbol] = price_df
        
        price_df = price_data_cache[symbol]

        # --- 2. Slice Price Data for the Trade Duration ---
        # Slicing with .loc is inclusive of both the start and end times.
        trade_candles = price_df.loc[trade['entry_time']:trade['exit_time']].copy()

        if trade_candles.empty:
            discrepancies.append({
                "trade": trade.to_dict(),
                "reason": "Data Slice Empty",
                "details": f"No price data found between {trade['entry_time']} and {trade['exit_time']}"
            })
            continue

        # --- 3. Verify Exit Reason ---
        exit_reason_valid = False
        exit_reason = trade['exit_reason']
        sl_price = trade['sl']
        tp_price = trade['tp']
        
        entry_candle = trade_candles.iloc[0]

        if IS_SHORT_STRATEGY:
            if exit_reason == 'SL_HIT_ON_ENTRY':
                if entry_candle['high'] >= sl_price:
                    exit_reason_valid = True
            elif exit_reason == 'SL_HIT':
                if (trade_candles['high'] >= sl_price).any():
                    exit_reason_valid = True
            elif exit_reason == 'TP_HIT':
                if (trade_candles['low'] <= tp_price).any():
                    exit_reason_valid = True
        else: # Logic for LONG trades
            # This section can be expanded for strategies that include long positions
            pass
        
        if not exit_reason_valid:
            details = f"Exit reason '{exit_reason}' not validated. "
            if 'SL' in exit_reason:
                details += f"Max high during trade was {trade_candles['high'].max():.4f}, SL was {sl_price:.4f}."
            elif 'TP' in exit_reason:
                details += f"Min low during trade was {trade_candles['low'].min():.4f}, TP was {tp_price:.4f}."
            discrepancies.append({
                "trade": trade.to_dict(),
                "reason": "Invalid Exit Reason",
                "details": details
            })

        # --- 4. Check for Premature Exits ---
        # We analyze all candles *before* the logged exit candle.
        # --- FIX for Premature Exit Logic (v1.4.0) ---
        # We skip this check for 'SL_HIT_ON_ENTRY' because the exit is expected
        # to happen on the entry candle itself, which would otherwise be flagged as premature.
        if exit_reason != 'SL_HIT_ON_ENTRY':
            candles_before_exit = trade_candles.iloc[:-1]
            
            if not candles_before_exit.empty:
                premature_sl_hit = (candles_before_exit['high'] >= sl_price).any()
                premature_tp_hit = (candles_before_exit['low'] <= tp_price).any()

                if premature_sl_hit or premature_tp_hit:
                    premature_sl_time = candles_before_exit[candles_before_exit['high'] >= sl_price].index.min()
                    premature_tp_time = candles_before_exit[candles_before_exit['low'] <= tp_price].index.min()
                    
                    details = "Potential premature exit. "
                    if pd.notna(premature_sl_time):
                        details += f"SL could have been hit at {premature_sl_time}. "
                    if pd.notna(premature_tp_time):
                        details += f"TP could have been hit at {premature_tp_time}."

                    discrepancies.append({
                        "trade": trade.to_dict(),
                        "reason": "Premature Exit Detected",
                        "details": details
                    })

        # --- 5. Recalculate and Verify PnL ---
        # This calculation assumes PnL is the simple price difference per share.
        # It may need adjustment for contracts, fees, or percentage-based PnL.
        if IS_SHORT_STRATEGY:
            recalculated_pnl = trade['entry_price'] - trade['exit_price']
        else: # PnL for a LONG trade
            recalculated_pnl = trade['exit_price'] - trade['entry_price']
        
        if not np.isclose(recalculated_pnl, trade['pnl'], atol=PNL_TOLERANCE):
            discrepancies.append({
                "trade": trade.to_dict(),
                "reason": "PnL Mismatch",
                "details": f"Logged PnL: {trade['pnl']:.4f}, Recalculated PnL: {recalculated_pnl:.4f}"
            })

    # --- 6. Report Results ---
    print("\n--- Audit Summary ---")
    print(f"Total Trades Audited: {total_trades}")

    if not discrepancies:
        print("✅ All trades passed the audit. No discrepancies found.")
        return

    print(f"🚨 Found {len(discrepancies)} total discrepancy instances.")
    
    # Count discrepancies by type for a high-level overview
    summary = {}
    for d in discrepancies:
        reason = d['reason']
        summary[reason] = summary.get(reason, 0) + 1
    
    print("\nDiscrepancy Breakdown:")
    for reason, count in sorted(summary.items()):
        print(f"- {reason}: {count} instance(s)")

    print("\n--- Detailed Discrepancies ---")
    
    # Group discrepancies by trade to present a clear, consolidated report per trade.
    grouped_discrepancies = {}
    for i, d in enumerate(discrepancies):
        # Use a unique identifier for each trade (entry_time)
        trade_index = d['trade']['entry_time']
        if trade_index not in grouped_discrepancies:
            grouped_discrepancies[trade_index] = {
                "trade_info": f"Trade for {d['trade']['symbol']} at {d['trade']['entry_time']}",
                "issues": []
            }
        grouped_discrepancies[trade_index]['issues'].append(f"[{d['reason']}] {d['details']}")

    for _, data in grouped_discrepancies.items():
        print(f"\n▶️  {data['trade_info']}:")
        for issue in data['issues']:
            print(f"  - {issue}")


# --- Script Execution ---
if __name__ == "__main__":
    audit_strategy(STRATEGY_NAME)
