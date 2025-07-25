Project Runbook & Operational Guide
This document provides the Standard Operating Procedure (SOP) for running the Nifty 200 Pullback Strategy's research pipeline.

Step 1: Data Acquisition
Run the Daily Equity Scraper: python fyers_equity_scraper.py

Run the Daily Index Scraper: python fyers_index_scraper.py

Run the 15-Minute Equity Scraper: python fyers_equity_scraper_15min.py

Step 2: Data Processing
Run the Indicator Calculator: python calculate_indicators_clean.py

Step 3: Generate "Golden Benchmarks" (Optional)
Generate Daily Benchmark: python final_backtester_benchmark_logger.py

Generate HTF Benchmark: python final_backtester_immediate_benchmark_htf.py (ensure timeframe is set correctly inside the script).

Step 4: Run Realistic Hybrid Backtests
Configure the Script: Open either final_backtester_v8_hybrid_optimized.py (for daily) or final_backtester_htf_hybrid.py (for weekly/monthly). Adjust the boolean toggles (True/False) in the config dictionary at the top of the file to test different combinations of rules.

Execute the Backtest:

For Daily: python final_backtester_v8_hybrid_optimized.py

For HTF: python final_backtester_htf_hybrid.py

Step 5: Analyze Feature Impact (Optional)
Run the Breakeven Impact Analyzer: After running a backtest with 'use_aggressive_breakeven': True, you can analyze its impact:

python analyze_breakeven_impact.py
