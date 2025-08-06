# strategy_optimizer.py
#
# Description:
# A script to automate the process of optimizing strategy filters. It
# iterates through a defined parameter space, runs the daily scanner and
# analyzer for each combination, and reports the best-performing configuration.

import pandas as pd
import os
import sys
import itertools
import importlib
import json

# To import functions from other scripts, ensure they are in the same directory
# and that their main execution block (if __name__ == "__main__":) is guarded.
daily_entry_scanner = importlib.import_module('daily_entry_scanner')
daily_signal_analyzer = importlib.import_module('daily_signal_analyzer')
daily_signal_characterizer = importlib.import_module('daily_signal_characterizer')

def run_single_backtest(scanner_config, analyzer_config, optimizer_log_path):
    """
    Executes a single backtest run with a given configuration.
    Returns the key performance metrics.
    """
    # Create a unique folder for this run's logs to avoid overwriting
    run_timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    run_log_folder = os.path.join(scanner_config['log_folder'], 'optimizer_runs', f"run_{run_timestamp}")
    os.makedirs(run_log_folder, exist_ok=True)

    scanner_config['log_folder'] = run_log_folder
    scanner_config['strategy_name'] = 'daily_entry_scanner'
    scanner_config['output_filename'] = 'entry_signals.csv'

    # Run the entry scanner with the current config
    sys.stdout.write(f"\n--- Running scanner for config: {json.dumps(scanner_config)} ---\n")
    daily_entry_scanner.run_scanner(scanner_config)

    # Check if signals were generated before proceeding
    scanner_output_path = os.path.join(run_log_folder, 'daily_entry_scanner', 'entry_signals.csv')
    if not os.path.exists(scanner_output_path):
        print("No signals generated. Skipping analysis.")
        return {'config': scanner_config, 'total_signals': 0, 'win_rate': 0.0, 'max_rr_achieved': 0.0}

    # Run the signal analyzer with the current config
    analyzer_config['log_folder'] = run_log_folder
    analyzer_config['scanner_strategy_name'] = 'daily_entry_scanner'
    analyzer_config['scanner_output_filename'] = 'entry_signals.csv'
    
    # Temporarily redirect stdout to capture the analyzer's report
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    daily_signal_analyzer.analyze_signals(analyzer_config)
    sys.stdout = old_stdout

    report_output = captured_output.getvalue()
    
    # Parse the report to get key metrics
    total_signals = 0
    win_rate = 0.0
    max_rr_achieved = 0.0

    lines = report_output.splitlines()
    for line in lines:
        if 'Total Valid Signals Analyzed' in line:
            total_signals = int(line.split(':')[1].strip())
        if 'Achieved >0R' in line:
            # The percentage is the 4th token from the end, after splitting by spaces
            win_rate = float(line.split()[-1][:-1]) # Get percentage and remove '%'
        if line.startswith('Achieved'):
            # This line finds the highest achieved R multiple
            try:
                rr = int(line.split('R')[0].split('>')[-1])
                max_rr_achieved = max(max_rr_achieved, rr)
            except ValueError:
                continue

    # Log the results of this run
    result = {
        'config': scanner_config,
        'total_signals': total_signals,
        'win_rate': win_rate,
        'max_rr_achieved': max_rr_achieved,
    }
    
    # Append the result to the optimizer log
    with open(optimizer_log_path, 'a') as f:
        f.write(json.dumps(result) + '\n')
    
    return result

def main():
    """Main function to run the strategy optimizer."""
    print("--- Starting Strategy Optimizer ---")

    # --- 1. Define the Parameter Search Space ---
    # Each list represents the values to test for a given parameter.
    # The script will run a backtest for every permutation of these lists.
    param_grid = {
        'use_ema_filter': [True, False],
        'ema_period': [10, 20, 50],
        'market_regime_filter': [True, False],
        'volume_filter': [True, False],
        'volume_multiplier': [1.0, 1.2, 1.5],
        'use_macd_range_filter': [True, False],
        'macd_hist_range': [(0.5, 3.0), (1.0, 5.0), (3.0, 7.0)] # Test different MACD ranges
    }

    # Use default configs from the existing scripts
    scanner_base_cfg = daily_entry_scanner.config
    analyzer_base_cfg = daily_signal_analyzer.config
    scanner_base_cfg['regime_ema_cross'] = None # Set EMA cross to None for consistency
    scanner_base_cfg['use_trend_momentum_filter'] = False # Ensure deprecated filter is off

    # Log file for all optimization results
    optimizer_log_file = 'optimization_results.jsonl'
    optimizer_log_path = os.path.join(scanner_base_cfg['log_folder'], optimizer_log_file)
    
    # Remove old log file to start fresh
    if os.path.exists(optimizer_log_path):
        os.remove(optimizer_log_path)
    
    all_runs = []
    
    # --- 2. Generate and Run All Permutations ---
    # Create all combinations of parameters
    param_names = sorted(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[name] for name in param_names)))
    
    print(f"Total number of backtest runs to perform: {len(combinations)}")
    
    for i, combination in enumerate(combinations):
        print(f"\n--- Running permutation {i+1}/{len(combinations)} ---")
        
        # Create a new config for this run by updating the base config
        current_scanner_cfg = scanner_base_cfg.copy()
        current_analyzer_cfg = analyzer_base_cfg.copy()
        
        for name, value in zip(param_names, combination):
            current_scanner_cfg[name] = value

        result = run_single_backtest(current_scanner_cfg, current_analyzer_cfg, optimizer_log_path)
        all_runs.append(result)
        
    print("\n--- Optimization Complete! ---")
    
    # --- 3. Analyze Results and Find the Best Performer ---
    results_df = pd.DataFrame(all_runs)
    
    if results_df.empty:
        print("No results to analyze.")
        return
        
    # Sort by a chosen metric (e.g., win_rate) to find the best configuration
    best_run = results_df.sort_values(by='win_rate', ascending=False).iloc[0]

    print("\n--- Best Performing Configuration ---")
    print(f"Win Rate: {best_run['win_rate']:.2f}%")
    print(f"Total Signals: {best_run['total_signals']}")
    print(f"Max R:R Achieved: {best_run['max_rr_achieved']}")
    print("\nConfiguration:")
    print(json.dumps(best_run['config'], indent=2))
    
    # You can also save the full results DataFrame for later analysis
    results_df.to_csv(os.path.join(scanner_base_cfg['log_folder'], 'all_optimization_results.csv'), index=False)

if __name__ == "__main__":
    # Add a try-except block to handle potential issues with missing scripts
    try:
        import io
        main()
    except (ImportError, FileNotFoundError) as e:
        print(f"Error: Could not import or find required files. Please ensure 'daily_entry_scanner.py' and 'daily_signal_analyzer.py' are in the same directory.")
        print(f"Original Error: {e}")
