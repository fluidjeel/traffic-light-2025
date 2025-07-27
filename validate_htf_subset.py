# validate_htf_subset.py
#
# Description:
# This script compares the output logs from the biased (benchmark) and
# unbiased (hybrid) HTF backtesters to validate that the unbiased results
# are a true subset of the biased universe. It confirms that the hybrid
# backtester does not invent setups that the core strategy logic did not find.
#
# MODIFICATION: Enhanced the final report to be a clear, standout summary
# block to prevent the pass/fail result from being missed in the console.
#
# How to Run:
# python validate_htf_subset.py <path_to_biased_all_setups.csv> <path_to_unbiased_all_setups.csv>

import pandas as pd
import sys
import os

def validate_subset(biased_log_path, unbiased_log_path):
    """
    Validates that the unbiased setups are a subset of the biased setups
    using the unique 'setup_id'.
    """
    print("--- Starting HTF Subset Validation ---")

    # 1. Check if log files exist
    if not os.path.exists(biased_log_path):
        print(f"\n❌ Error: Biased log file not found at '{biased_log_path}'")
        return
    if not os.path.exists(unbiased_log_path):
        print(f"\n❌ Error: Unbiased log file not found at '{unbiased_log_path}'")
        return

    # 2. Load the log files into pandas DataFrames
    try:
        df_biased = pd.read_csv(biased_log_path)
        df_unbiased = pd.read_csv(unbiased_log_path)
        print("\nSuccessfully loaded both log files.")
    except Exception as e:
        print(f"\n❌ Error reading CSV files: {e}")
        return

    # 3. Ensure the critical 'setup_id' column exists in both logs
    if 'setup_id' not in df_biased.columns or 'setup_id' not in df_unbiased.columns:
        print("\n❌ Error: 'setup_id' column not found in one or both logs.")
        print("Please ensure you are using the updated backtesting scripts with enhanced logging.")
        return

    # 4. Get the unique set of setup_ids from each log
    biased_ids = set(df_biased['setup_id'].unique())
    unbiased_ids = set(df_unbiased['setup_id'].unique())

    # 5. Perform the core validation: check if the unbiased set is a subset of the biased set
    is_subset = unbiased_ids.issubset(biased_ids)
    
    # 6. Report the results clearly
    print("\n--- ANALYSIS COMPLETE ---")
    
    discrepancy_ids = unbiased_ids - biased_ids
    missed_by_unbiased = biased_ids - unbiased_ids

    # --- NEW: Final Summary Block ---
    print("\n" + "="*50)
    print(" " * 15 + "VALIDATION SUMMARY")
    print("="*50)
    
    if is_subset:
        print("\n✅ STATUS: PASSED")
        print("   The unbiased (hybrid) log is a true subset of the biased log.")
    else:
        print("\n❌ STATUS: FAILED")
        print("   The unbiased log is NOT a subset of the biased log.")
        print(f"\n   Discrepancies Found: {len(discrepancy_ids)}")
        print(f"   These setups exist in the UNBIASED log but not the BIASED one:")
        for an_id in sorted(list(discrepancy_ids))[:10]: # Print top 10
            print(f"     - {an_id}")

    print("\n--- STATISTICS ---")
    print(f"Total Unique Setups in Biased (Benchmark) Log: {len(biased_ids)}")
    print(f"Total Unique Setups in Unbiased (Hybrid) Log:  {len(unbiased_ids)}")
    
    if missed_by_unbiased:
        print(f"\nCorrectly Filtered/Not Triggered by Hybrid: {len(missed_by_unbiased)}")
    
    print("\n" + "="*50)
    print("\n--- Validation Complete ---")


if __name__ == "__main__":
    # This allows the script to be run from the command line with file paths as arguments.
    if len(sys.argv) != 3:
        print("\nUsage: python validate_htf_subset.py <path_to_biased_all_setups.csv> <path_to_unbiased_all_setups.csv>")
        print("Example: python validate_htf_subset.py backtest_logs/biased.csv backtest_logs/unbiased.csv")
        sys.exit(1)
    
    biased_file = sys.argv[1]
    unbiased_file = sys.argv[2]
    
    validate_subset(biased_file, unbiased_file)
