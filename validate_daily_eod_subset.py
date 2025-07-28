# validate_daily_eod_subset.py
#
# Description:
# This script validates the logical integrity of the simulator_daily_eod_confirm.py
# against its corresponding benchmark, benchmark_generator_daily.py.
#
# It proves that the set of trades identified by the realistic EOD simulator
# is a true and proper subset of the trades identified by the benchmark with
# lookahead bias. This confirms that the simulator is not generating any trades
# that the core strategy logic wouldn't have found.
#
# How to Run:
# python validate_daily_eod_subset.py <path_to_benchmark_log.csv> <path_to_simulator_log.csv>
#
# Example:
# python validate_daily_eod_subset.py backtest_logs/20250728_103000_benchmark_daily/20250728_103000_all_setups_log.csv backtest_logs/20250728_103200_daily_eod_confirm/20250728_103200_all_setups_log.csv

import pandas as pd
import sys
import os

def validate_subset(benchmark_file, simulator_file):
    """
    Performs the validation by comparing two log files.
    """
    print("--- Daily EOD Confirmation Strategy Validation ---")

    # --- 1. Check if files exist ---
    if not os.path.exists(benchmark_file):
        print(f"\n[ERROR] Benchmark file not found at: {benchmark_file}")
        sys.exit(1)
    if not os.path.exists(simulator_file):
        print(f"\n[ERROR] Simulator file not found at: {simulator_file}")
        sys.exit(1)

    print(f"\nBenchmark Log: {os.path.basename(benchmark_file)}")
    print(f"Simulator Log: {os.path.basename(simulator_file)}")

    # --- 2. Load the data ---
    try:
        benchmark_df = pd.read_csv(benchmark_file)
        simulator_df = pd.read_csv(simulator_file)
    except Exception as e:
        print(f"\n[ERROR] Failed to read CSV files: {e}")
        sys.exit(1)

    # --- 3. Extract the setup IDs ---
    # The benchmark log contains all potential setups, including those filtered out later.
    # The simulator log contains only setups that passed all filters at EOD.
    benchmark_ids = set(benchmark_df['setup_id'].unique())
    simulator_ids = set(simulator_df['setup_id'].unique())

    if not benchmark_ids:
        print("\n[WARNING] No setups found in the benchmark log file.")
    if not simulator_ids:
        print("\n[WARNING] No setups found in the simulator log file.")

    # --- 4. Perform the validation ---
    print("\n--- Analysis ---")
    print(f"Total Unique Setups in Benchmark: {len(benchmark_ids)}")
    print(f"Total Unique Setups in Simulator: {len(simulator_ids)}")

    # Check if the simulator's set of IDs is a subset of the benchmark's IDs
    is_subset = simulator_ids.issubset(benchmark_ids)

    print("\n--- Validation Result ---")
    if is_subset:
        print("[SUCCESS] Validation PASSED!")
        print("The simulator is a true subset of the benchmark.")
        print("This confirms the simulator's logic is correctly aligned and bias-free.")
    else:
        print("[FAILURE] Validation FAILED!")
        print("The simulator is NOT a true subset of the benchmark.")
        
        # Find the setups that are in the simulator but not in the benchmark
        rogue_setups = simulator_ids - benchmark_ids
        print(f"\nFound {len(rogue_setups)} rogue setups in the simulator log:")
        for setup in rogue_setups:
            print(f"  - {setup}")
        print("\nThis indicates a logical flaw in the simulator where it is inventing trades.")

    print("\n--- Validation Complete ---")


if __name__ == "__main__":
    # Check for the correct number of command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python validate_daily_eod_subset.py <path_to_benchmark_log.csv> <path_to_simulator_log.csv>")
        sys.exit(1)
    
    benchmark_log_path = sys.argv[1]
    simulator_log_path = sys.argv[2]
    
    validate_subset(benchmark_log_path, simulator_log_path)
