Project Code Explained
This document provides a functional overview of the key Python scripts in the Nifty 200 Pullback Strategy project.

1. Data Pipeline Scripts
fyers_equity_scraper.py & fyers_equity_scraper_15min.py: Primary data acquisition tools for all stocks.

fyers_index_scraper.py: Specialized scraper for daily index data (NIFTY200_INDEX, INDIAVIX).

calculate_indicators_clean.py: The definitive data processing engine. It takes all raw daily data, creates all higher timeframes, calculates all indicators (including volume_20_sma for HTFs), and saves the final, clean files.

2. Backtesting Engine Scripts
2.1. Benchmark Generators (Biased)
final_backtester_benchmark_logger.py: Generates the "Golden Benchmark" for the daily strategy using lookahead bias.

final_backtester_immediate_benchmark_htf.py: Generates the "Golden Benchmark" for the HTF-immediate strategy using lookahead bias.

2.2. Hybrid Backtesters (Bias-Free)
final_backtester_v8_hybrid_optimized.py: The state-of-the-art, realistic backtesting engine for the daily strategy. It identifies a setup on Day T-1 and executes on Day T using real-time intraday data.

final_backtester_htf_hybrid.py: A completely redesigned, from-scratch backtester for higher timeframes built on the "Scout and Sniper" principle to guarantee the elimination of lookahead bias.

Scout Logic (EOD): At the end of Day T, it finds stocks that have a valid weekly pullback pattern AND had a price breakout on the completed daily candle.

Sniper Logic (Intraday): On Day T+1, it monitors only the scouted stocks and waits for real-time conviction (volume, market strength) to enter.

2.3. Shared Features in Hybrid Backtesters
Both bias-free backtesters are equipped with a suite of advanced, toggleable features:

Dynamic Slippage Model: Simulates realistic fills.

Toggleable Entry Filters: cancel_on_gap_up, prevent_entry_below_trigger.

Toggleable Trade Management: use_partial_profit_leg, use_aggressive_breakeven, trail_on_htf_low_after_be.

3. Analysis Scripts
analyze_breakeven_impact.py: A standalone script that consumes the trades_detail.csv log to provide a quantitative analysis of the net P&L impact of the "Aggressive Breakeven" feature.