Master Project Context: Traffic Light 2025 (Version 5.1)
To the Large Language Model:

This is a private, custom algorithmic trading strategy. The information contained in the attached documents and code files is the single source of truth. Do not reference any external code, public GitHub repositories, or general trading knowledge. Your sole objective is to build a complete and accurate mental model of this project based only on the provided materials.

Public GitHub Repo for context: https://github.com/fluidjeel/traffic-light-2025

Your tasks are as follows:

Ingest Context: Fully ingest the context provided below and in the specified file sequence.

Prove Foundational Understanding: Your first response must be a direct paraphrase of the rules from the "Definitive Trading Strategy Logic" section and a summary of the key architectural components. Do not introduce any external concepts.

Prove Deeper Understanding: After your initial summary, you must answer the following questions to prove you have understood the relationship between the system's components:

Question 1: In monthly_tfl_simulator.py, if a researcher sets 'use_universal_pipeline': False in the data_pipeline_config, which specific folder paths will the script use for its data_folder_base and intraday_data_folder, and what will be the exact filename it looks for to load the intraday Nifty 200 Index data?

Question 2: Explain the critical difference between using a cash_at_sod variable (defined once per day) versus the real-time self.portfolio['cash'] variable for calculating capital_to_risk inside the execute_entry function. Why was this change essential for the integrity of the backtest?

Context Setting Starts
Objective:
The following document provides a comprehensive, explicit, and highly contextualized overview of a Python-based algorithmic trading project. Your goal is to fully ingest this document and the specified files in the correct sequence to build a complete and accurate mental model of the project's architecture, data flow, and the precise logic of its trading strategies.

Section 1: Context Setting - The Required Reading Sequence
To understand this project, you must study the following files in the precise order listed below. This sequence is designed to build your knowledge from the high-level concept down to the specific implementation details.

project_implementation.md: Read this file first and in its entirety. It contains the master overview of the project's goals, architecture, and the definitive logic for all trading strategies.

project_code_explained.md: After understanding the master plan, read this file to get a high-level overview of what each specific Python script accomplishes.

project_runbook_operational_guide.md: This document will explain the standard operating procedure for running the entire research pipeline, from data acquisition to analysis.

Core Universal Pipeline & Analysis Scripts: Finally, review the source code of the key scripts to understand the implementation details.

universal_fyers_scraper.py (The universal data scraper)

universal_calculate_indicators.py (The universal indicator calculator)

daily_tfl_simulator.py (The daily timeframe simulator)

weekly_tfl_simulator.py (The weekly timeframe simulator)

monthly_tfl_simulator.py (The monthly timeframe simulator)

mae_analyzer_percent.py (The stop-loss analysis tool)

Section 2: High-Level Project Goal & Philosophy
2.1. Goal
The primary objective is to develop, backtest, and automate a profitable, long-only, swing trading strategy with a verifiable and realistic edge, applied to the Nifty 200 stock universe across multiple timeframes.

2.2. Core Philosophy
Our guiding principle is an unwavering commitment to eliminating all forms of bias. All development and testing must be conducted using the final, corrected simulators that have been rigorously audited to remove lookahead biases and critical calculation flaws.

Section 3: System Architecture & The Universal Data Pipeline
The project is a streamlined, modular pipeline designed for rigorous, bias-free research.

Data Acquisition: The universal_fyers_scraper.py script is the single entry point for all data. It intelligently downloads and updates raw daily and 15-minute historical data for all stocks and indices.

Data Processing: The universal_calculate_indicators.py script acts as the central data processing engine. It reads the raw daily data, resamples it into higher timeframes, calculates all necessary indicators, and saves these clean, enriched data files.

Realistic Simulation: The *_tfl_simulator.py scripts run the strategies in a bias-free manner. They have been fully corrected to eliminate all known lookahead biases and calculation flaws.

Post-Hoc Analysis: The mae_analyzer_percent.py script ingests the detailed trade logs to perform "what-if" simulations for data-driven stop-loss optimization.

Section 4: The Definitive and Exclusive Trading Strategy Logic
The core of the "Traffic Light 2025" strategy is a pullback continuation pattern. The general logic is to identify a stock in an established uptrend that has experienced a brief pause or "pullback" (represented by one or more red candles) and then resumed its upward momentum (represented by a green candle).

Universal Trade Management Rules (Corrected & Bias-Free):

Position Sizing: All simulators correctly calculate position size based on the real-time available cash balance, preventing the leveraging of unrealized profits.

VIX Data Timing: All intraday decisions correctly use the VIX closing value from the previous trading day (T-1), ensuring no lookahead bias.

Stop-Loss: Can be configured to use a fixed percentage or a lookback to the lowest low of recent candles.

Trailing Stop: All simulators use a percentage-based buffer for the aggressive breakeven logic to ensure it scales correctly with stock price.

Section 5: Current Project Status & Immediate Next Steps
Current Status: The project has successfully migrated to a unified, streamlined data pipeline. The suite of simulators has been rigorously audited and corrected, eliminating all known lookahead biases and critical calculation flaws. The project is now in a robust state, ready for methodical strategy research and optimization.

Immediate Next Steps: The focus now shifts from fixing flaws to stress-testing and refining the strategies by methodically enabling the various conviction and regime filters within the simulator configurations.