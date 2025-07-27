# validate_daily_subset.py
#
# Description:
# This script compares the output logs from the daily benchmark generator and
# the daily hybrid simulator to validate that the simulator's results
# are a true subset of the benchmark's universe.
#
# How to Run:
# python validate_daily_subset.py <path_to_benchmark_daily_log.csv> <path_to_simulator_daily_log.csv>

import pandas as pd
import sys
import os

def validate_subset(benchmark_log_path, simulator_log_path):
    """
    Validates that the simulator setups are a subset of the benchmark setups.
    """
    print("--- Starting Daily Subset Validation ---")

    if not os.path.exists(benchmark_log_path):
        print(f"\n❌ Error: Benchmark log file not found at '{benchmark_log_path}'")
        return
    if not os.path.exists(simulator_log_path):
        print(f"\n❌ Error: Simulator log file not found at '{simulator_log_path}'")
        return

    try:
        df_benchmark = pd.read_csv(benchmark_log_path)
        df_simulator = pd.read_csv(simulator_log_path)
        print("\nSuccessfully loaded both log files.")
    except Exception as e:
        print(f"\n❌ Error reading CSV files: {e}")
        return

    if 'setup_id' not in df_benchmark.columns or 'setup_id' not in df_simulator.columns:
        print("\n❌ Error: 'setup_id' column not found in one or both logs.")
        return

    # For daily validation, we only care about the setups that were IDENTIFIED
    # before any filtering occurred on the trigger day.
    benchmark_ids = set(df_benchmark['setup_id'].unique())
    simulator_ids = set(df_simulator[df_simulator['status'] == 'IDENTIFIED']['setup_id'].unique())

    is_subset = simulator_ids.issubset(benchmark_ids)
    
    print("\n--- ANALYSIS COMPLETE ---")
    
    discrepancy_ids = simulator_ids - benchmark_ids
    missed_by_simulator = benchmark_ids - simulator_ids

    print("\n" + "="*50)
    print(" " * 15 + "VALIDATION SUMMARY (DAILY)")
    print("="*50)
    
    if is_subset:
        print("\n✅ STATUS: PASSED")
        print("   The daily simulator's watchlist is a true subset of the benchmark's identified setups.")
    else:
        print("\n❌ STATUS: FAILED")
        print("   The daily simulator's watchlist is NOT a subset of the benchmark's setups.")
        print(f"\n   Discrepancies Found: {len(discrepancy_ids)}")
        print(f"   These setups exist in the SIMULATOR log but not the BENCHMARK one:")
        for an_id in sorted(list(discrepancy_ids))[:10]:
            print(f"     - {an_id}")

    print("\n--- STATISTICS ---")
    print(f"Total Unique Setups in Benchmark Log:    {len(benchmark_ids)}")
    print(f"Total Unique Setups in Simulator Log:    {len(simulator_ids)}")
    
    if missed_by_simulator:
        print(f"\nSetups found by Benchmark but not Simulator: {len(missed_by_simulator)}")
    
    print("\n" + "="*50)
    print("\n--- Validation Complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("\nUsage: python validate_daily_subset.py <path_to_benchmark_daily.csv> <path_to_simulator_daily.csv>")
        sys.exit(1)
    
    benchmark_file = sys.argv[1]
    simulator_file = sys.argv[2]
    
    validate_subset(benchmark_file, simulator_file)
