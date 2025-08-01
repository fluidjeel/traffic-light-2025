# run_data_pipeline.py
#
# Description:
# This is a wrapper script to automate the entire data pipeline workflow.
# It runs all individual data scrapers in the correct sequence, followed by
# the indicator calculation script, to ensure a complete and up-to-date dataset
# is available for the backtesting simulators.
#
# Usage:
#   python run_data_pipeline.py

import subprocess
import sys
import os

def run_script(script_name, interactive=False):
    """
    Executes a Python script and checks for errors.
    If interactive is True, the output is streamed directly to the console.
    """
    print(f"\n========================================================")
    print(f"| Running {script_name}...")
    print(f"========================================================\n")
    
    try:
        if interactive:
            # For interactive scripts, pipe stdout and stdin directly
            result = subprocess.run(
                [sys.executable, script_name],
                check=True
            )
        else:
            # For non-interactive scripts, capture output
            result = subprocess.run(
                [sys.executable, script_name],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        
        print(f"\n>>> Successfully completed {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        if not interactive:
            print(e.stdout)
            print(e.stderr)
        print(f"\n!!! ERROR: {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n!!! ERROR: {script_name} not found. Please ensure it is in the same directory.")
        return False

def main():
    """
    Main function to orchestrate the entire data pipeline.
    """
    scripts = [
        ("fyers_equity_scraper.py", True),
        ("fyers_index_scraper.py", True),
        ("fyers_equity_scraper_15min.py", True),
        ("fyers_nifty200_index_scraper.py", True),
        ("calculate_indicators_clean.py", False)
    ]

    print("--- Starting Automated Data Pipeline ---")
    
    for script, is_interactive in scripts:
        if not run_script(script, is_interactive):
            print("\n!!! Data pipeline aborted due to a critical error. Please fix the issue and run again.")
            sys.exit(1)
    
    print("\n\n========================================================")
    print("| All data processing and indicator calculations are complete!")
    print("| Your simulators are now ready to run with fresh data.")
    print("========================================================\n")

if __name__ == "__main__":
    # Ensure the script is run from the correct directory
    if os.path.basename(os.getcwd()) != 'algo-2025':
        print("Please run this script from the 'algo-2025' directory.")
        sys.exit(1)
        
    main()
