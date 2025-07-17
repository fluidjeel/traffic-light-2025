import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

def run_backtest(filepath: str, symbol: str, exit_logic: str, rr_target: float = 2.5):
    """
    Runs a daily backtest for a single stock using one of the two specified exit strategies.

    Args:
        filepath (str): Path to the stock's CSV data file.
        symbol (str): The stock symbol for labeling.
        exit_logic (str): The exit strategy to use. Must be 'simplified_ts' or 'fixed_rr_exit'.
        rr_target (float): The risk-reward target for the 'fixed_rr_exit' logic.
    """
    # --- 1. Data Loading and Preparation ---
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if 'datetime' in df.columns: df.rename(columns={'datetime': 'date'}, inplace=True)
        for col in df.columns:
            if col.upper().startswith('EMA_30'): df.rename(columns={col: 'ema_30'}, inplace=True)
        
        df = df.loc[:,~df.columns.duplicated()]
        df['date'] = pd.to_datetime(df['date'])
        required_cols = ['date', 'open', 'high', 'low', 'close', 'ema_30']
        if not all(col in df.columns for col in required_cols): return None
    except Exception:
        return None
    
    df.dropna(subset=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    trades = []
    in_position = False
    position_details = {}

    # --- 2. Main Backtesting Loop ---
    for i in range(5, len(df)):
        current_candle = df.iloc[i]
        
        if in_position:
            # --- Position Management ---
            if current_candle['low'] <= position_details['stop_loss']:
                pnl = (position_details['stop_loss'] - position_details['entry_price']) * 2
                trades.append({'symbol': symbol, 'pnl': pnl})
                in_position = False; continue

            if not position_details['partial_exit_achieved'] and current_candle['high'] >= position_details['target_price_1_1']:
                trades.append({'symbol': symbol, 'pnl': position_details['risk']})
                position_details['partial_exit_achieved'] = True
            
            if position_details['partial_exit_achieved']:
                exit_triggered = False
                final_exit_price = 0

                if exit_logic == 'simplified_ts':
                    prev_candle = df.iloc[i-1]
                    if prev_candle['close'] > position_details['entry_price'] and prev_candle['close'] > prev_candle['open']:
                        position_details['stop_loss'] = max(position_details['stop_loss'], prev_candle['low'])
                
                elif exit_logic == 'fixed_rr_exit':
                    if current_candle['high'] >= position_details['final_target_price']:
                        exit_triggered = True
                        final_exit_price = position_details['final_target_price']

                if exit_triggered:
                    pl = final_exit_price - position_details['entry_price']
                    trades.append({'symbol': symbol, 'pnl': pl})
                    in_position = False
        else:
            # --- Entry Scanning ---
            candle_minus_1 = df.iloc[i-1]
            candle_minus_2 = df.iloc[i-2]
            if (candle_minus_1['close'] < candle_minus_1['open']) and (candle_minus_1['close'] > candle_minus_1['ema_30']):
                entry_trigger_price = max(candle_minus_1['high'], candle_minus_2['high'])
                if current_candle['open'] < entry_trigger_price and current_candle['high'] >= entry_trigger_price:
                    entry_price = entry_trigger_price
                    stop_loss = df['low'].iloc[i-5:i].min()
                    risk = entry_price - stop_loss
                    if risk <= 0: continue
                    in_position = True
                    position_details = {
                        'entry_price': entry_price, 'stop_loss': stop_loss, 'risk': risk,
                        'target_price_1_1': entry_price + risk,
                        'final_target_price': entry_price + (risk * rr_target),
                        'partial_exit_achieved': False
                    }
    
    if not trades: return None
    trades_df = pd.DataFrame(trades)
    trade_summary = trades_df.groupby(np.arange(len(trades_df)) // 2).agg(
        symbol=('symbol', 'first'),
        pnl_points_total=('pnl', 'sum')
    ).reset_index(drop=True)
    trade_summary['outcome'] = np.where(trade_summary['pnl_points_total'] > 0, 'Win', 'Loss')
    return trade_summary

# ========================================================================================
# ## --- SCRIPT CONFIGURATION --- ##
# ========================================================================================
DATA_FOLDER = 'daily_with_indicators'
NIFTY_200_FILE = 'nifty200.csv'
# The two strategies to be compared
EXIT_STRATEGIES = ['simplified_ts', 'fixed_rr_exit']
# ========================================================================================

# --- Main Execution Block ---
if __name__ == "__main__":
    all_results = []
    
    try:
        nifty200_df = pd.read_csv(NIFTY_200_FILE)
        stocks_to_test = nifty200_df['Symbol'].tolist()
        print(f"✅ Found {len(stocks_to_test)} stocks in {NIFTY_200_FILE}.")
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: '{NIFTY_200_FILE}' not found. Please place it in the same directory.")
        exit()

    print("🚀 Starting strategy comparison backtest...")
    print("-" * 50)

    for logic in EXIT_STRATEGIES:
        print(f"📊 Running tests for strategy: '{logic.upper()}'")
        strategy_summaries = []
        
        processed_count = 0
        for symbol in stocks_to_test:
            filepath = os.path.join(DATA_FOLDER, f"{symbol}_daily_with_indicators.csv")
            
            summary_df = run_backtest(filepath=filepath, symbol=symbol, exit_logic=logic)
            if summary_df is not None:
                strategy_summaries.append(summary_df)
                processed_count += 1
        
        print(f"    -> Processed {processed_count} stocks with available data for this strategy.")

        if strategy_summaries:
            report = pd.concat(strategy_summaries, ignore_index=True)
            performance = report.groupby('symbol').agg(
                total_pnl=('pnl_points_total', 'sum'),
                win_rate_percent=('outcome', lambda x: (x == 'Win').sum() / len(x) * 100 if len(x) > 0 else 0)
            ).reset_index()
            performance['logic'] = logic
            all_results.append(performance)

    print("-" * 50)
    print("✅ All backtests complete. Generating final report...")

    # --- Final Comparison Report ---
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        pnl_pivot = pd.pivot_table(final_df, values='total_pnl', index='symbol', columns='logic')
        
        print("\n\n" + "="*60)
        print("              STRATEGY COMPARISON (TOTAL P/L)")
        print("="*60)
        print(pnl_pivot.to_string())
        
        print("\n\n" + "="*60)
        print("              OVERALL PERFORMANCE (TOTAL P/L)")
        print("="*60)
        overall_summary = pnl_pivot.sum().sort_values(ascending=False)
        print(overall_summary.to_string())

        report_filename = 'strategy_comparison_report.csv'
        pnl_pivot.to_csv(report_filename)
        print(f"\n\n💾 Final detailed report saved to: {report_filename}")

    else:
        print("\n❌ No trades were executed for any stock with any strategy.")