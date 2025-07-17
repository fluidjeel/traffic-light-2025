import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

# Suppress common warnings from pandas for a cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

def run_portfolio_backtest(initial_capital: float, risk_per_trade_percent: float, stocks_to_test: list, data_folder: str):
    """
    Runs a portfolio-level backtest with a rewritten, accuracy-focused simulation engine.
    This version prioritizes correct P&L and CAGR calculation over speed.
    """
    print("⏳ Loading and consolidating all stock data...")
    all_data = []
    # Loop through the stock list and load data if the file exists
    for symbol in stocks_to_test:
        filepath = os.path.join(data_folder, f"{symbol}_daily_with_indicators.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # Standardize column names for consistency
                if 'datetime' in df.columns: df.rename(columns={'datetime': 'date'}, inplace=True)
                for col in df.columns:
                    if col.upper().startswith('EMA_30'): df.rename(columns={col: 'ema_30'}, inplace=True)
                # Remove duplicate columns that might result from renaming
                df = df.loc[:,~df.columns.duplicated()]
                df['date'] = pd.to_datetime(df['date'])
                df['symbol'] = symbol
                all_data.append(df)
            except Exception:
                continue

    if not all_data:
        print("❌ No valid data files found to run the backtest.")
        return None, None

    # Create a single master dataframe sorted by date
    master_df = pd.concat(all_data, ignore_index=True).sort_values(by='date').reset_index(drop=True)
    master_df.dropna(subset=['ema_30', 'open', 'high', 'low', 'close'], inplace=True)
    
    # Pre-group data by symbol for efficient lookups
    print("⏳ Optimizing data by pre-grouping symbols...")
    stock_data_groups = {symbol: group.reset_index(drop=True) for symbol, group in master_df.groupby('symbol')}
    
    # Portfolio and logging setup
    equity_curve = pd.DataFrame({'date': [master_df['date'].min()], 'equity': [initial_capital]})
    trades_log = []
    in_position = {} 

    unique_dates = master_df['date'].unique()
    print(f"✅ Data loaded and optimized. Starting backtest across {len(unique_dates)} trading days...")

    # Main backtesting loop iterates through each unique day
    for day_index, date in enumerate(unique_dates):
        
        # --- DEBUG: Print continuous progress ---
        print(f"  -> Processing Day {day_index + 1}/{len(unique_dates)}: {pd.to_datetime(date).strftime('%Y-%m-%d')}")

        equity_at_start_of_day = equity_curve['equity'].iloc[-1]
        todays_pnl = 0

        # 1. Manage existing positions for the current day
        for symbol in list(in_position.keys()):
            # Get the full history for this stock to find the current candle
            stock_history = stock_data_groups[symbol]
            current_day_df = stock_history[stock_history['date'] == date]
            
            if not current_day_df.empty:
                current_candle = current_day_df.iloc[0]
                details = in_position[symbol]

                # Check for Stop-Loss
                if current_candle['low'] <= details['stop_loss']:
                    pnl = (details['stop_loss'] - details['entry_price']) * details['shares']
                    todays_pnl += pnl
                    trades_log.append({'symbol': symbol, 'entry_date': details['entry_date'], 'exit_date': date, 'pnl': pnl, 'exit_type': 'Stop-Loss'})
                    print(f"      ❌ EXIT: {symbol} stopped out. P&L: {pnl:,.2f}")
                    del in_position[symbol]
                    continue
                
                # Check for 1:1 Partial Profit
                if not details['partial_exit_achieved'] and current_candle['high'] >= details['target_price_1_1']:
                    exit_price = details['target_price_1_1']
                    pnl = (exit_price - details['entry_price']) * (details['shares'] / 2)
                    todays_pnl += pnl
                    trades_log.append({'symbol': symbol, 'entry_date': details['entry_date'], 'exit_date': date, 'pnl': pnl, 'exit_type': 'Partial Profit (1:1)'})
                    print(f"      💰 PARTIAL EXIT: {symbol} hit 1:1 target. P&L: {pnl:,.2f}")
                    details['shares'] /= 2
                    details['partial_exit_achieved'] = True

                # Trail the Stop-Loss
                prev_candle = details['last_candle']
                if prev_candle['close'] > details['entry_price'] and prev_candle['close'] > prev_candle['open']:
                    details['stop_loss'] = max(details['stop_loss'], prev_candle['low'])
                
                in_position[symbol]['last_candle'] = current_candle

        # 2. Scan for new entries for the current day
        daily_data = master_df[master_df['date'] == date]
        for _, current_candle in daily_data.iterrows():
            symbol = current_candle['symbol']
            if symbol not in in_position:
                stock_history = stock_data_groups[symbol]
                # Get history up to the day *before* the current day for signal generation
                signal_period_data = stock_history[stock_history['date'] < date]

                if len(signal_period_data) < 5: continue
                
                candle_minus_1 = signal_period_data.iloc[-1]
                candle_minus_2 = signal_period_data.iloc[-2]

                if (candle_minus_2['close'] < candle_minus_2['open']) and \
                   (candle_minus_1['close'] > candle_minus_1['open']) and \
                   (candle_minus_1['close'] > candle_minus_1['ema_30']):
                    
                    entry_trigger_price = max(candle_minus_1['high'], candle_minus_2['high'])
                    if current_candle['open'] < entry_trigger_price and current_candle['high'] >= entry_trigger_price:
                        entry_price = entry_trigger_price
                        stop_loss = signal_period_data['low'].tail(5).min()
                        risk_per_share = entry_price - stop_loss
                        if risk_per_share <= 0: continue
                        
                        risk_amount = equity_at_start_of_day * (risk_per_trade_percent / 100)
                        shares = int(risk_amount / risk_per_share)
                        if shares == 0: continue

                        print(f"      ✅ ENTRY: {symbol} triggered at {entry_price:.2f}. SL: {stop_loss:.2f}")
                        in_position[symbol] = {
                            'entry_price': entry_price, 'stop_loss': stop_loss, 'shares': shares,
                            'target_price_1_1': entry_price + risk_per_share,
                            'partial_exit_achieved': False, 'entry_date': date,
                            'last_candle': current_candle
                        }
        
        # Update equity curve once at the end of the day
        final_daily_equity = equity_at_start_of_day + todays_pnl
        equity_curve = pd.concat([equity_curve, pd.DataFrame([{'date': date, 'equity': final_daily_equity}])], ignore_index=True)

    print("\n✅ Backtest Complete.")
    return equity_curve, pd.DataFrame(trades_log)

# ========================================================================================
# ## --- SCRIPT CONFIGURATION --- ##
# ========================================================================================
DATA_FOLDER = 'daily_with_indicators'
NIFTY_200_FILE = 'nifty200.csv'
INITIAL_CAPITAL = 1000000.0
RISK_PER_TRADE_PERCENT = 1.0
OUTPUT_FOLDER = 'backtest_logs'
# ========================================================================================

if __name__ == "__main__":
    try:
        nifty200_df = pd.read_csv(NIFTY_200_FILE)
        stocks_to_test = nifty200_df['Symbol'].tolist()
        print(f"✅ Found {len(stocks_to_test)} stocks in {NIFTY_200_FILE}.")
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: '{NIFTY_200_FILE}' not found. Please place it in the same directory.")
        exit()

    equity_curve, trades_df = run_portfolio_backtest(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade_percent=RISK_PER_TRADE_PERCENT,
        stocks_to_test=stocks_to_test,
        data_folder=DATA_FOLDER
    )

    if equity_curve is not None and not equity_curve.empty:
        # --- Final Report Calculation ---
        final_equity = equity_curve['equity'].iloc[-1]
        total_return_percent = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        start_date = equity_curve['date'].iloc[0]
        end_date = equity_curve['date'].iloc[-1]
        years = (end_date - start_date).days / 365.25 if pd.notna(start_date) and pd.notna(end_date) else 0
        cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

        summary_text = (
            "==================================================\n"
            "           PORTFOLIO PERFORMANCE SUMMARY\n"
            "==================================================\n"
            f"Initial Capital:       Rs.{INITIAL_CAPITAL:,.2f}\n"
            f"Final Capital:         Rs.{final_equity:,.2f}\n"
            f"Net Profit/Loss:       Rs.{final_equity - INITIAL_CAPITAL:,.2f}\n"
            f"Total Return:          {total_return_percent:.2f}%\n"
            f"CAGR:                  {cagr:.2f}%\n"
            f"Backtest Period:       {years:.2f} years\n"
            "==================================================\n"
        )
        print("\n" + summary_text)

        # --- Save results to timestamped files ---
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        summary_filename = os.path.join(OUTPUT_FOLDER, f"{timestamp}_summary_report.txt")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"💾 Summary report saved to '{summary_filename}'")
        
        if trades_df is not None and not trades_df.empty:
            trades_filename = os.path.join(OUTPUT_FOLDER, f"{timestamp}_trades_log.csv")
            trades_df.to_csv(trades_filename, index=False)
            print(f"💾 Detailed trade log saved to '{trades_filename}'")
    else:
        print("\n❌ No trades were executed, no report generated.")
