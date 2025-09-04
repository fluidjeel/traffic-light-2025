import pandas as pd
import os
import numpy as np
import datetime
from pytz import timezone
import warnings
import matplotlib.pyplot as plt
import sys
import codecs # Import codecs for the fix

# SCRIPT VERSION v3.1 (Unified Simulator - Final Patched)
#
# ARCHITECTURAL NOTE:
# This is the definitive, complete version of the unified backtesting engine.
# It replaces the previous scaffolded version with the full, unabridged, and
# perfectly symmetrical logic ported from the hardened v3.7 long simulator and
# its symmetrical short counterpart.
#
# NEW FEATURE:
# v3.1: - Significantly enhanced the `calculate_and_log_metrics` function to
#         provide a professional-grade analytics summary.
#
# PREVIOUS VERSION:
# v3.0: - Added a trend-agnostic mode for advanced testing.

# Fix for pandas_ta compatibility
np.NaN = np.nan
import pandas_ta as ta

# --- SUPPRESS FUTUREWARNING ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- CONFIGURATION SETTINGS ---
# ==============================================================================

# --- Backtest Period ---
START_DATE = '2018-01-01'
END_DATE = '2025-08-22'

# --- Portfolio & Risk Management ---
STRATEGY_NAME = "TrafficLight-Manny-UNIFIED_PORTFOLIO"
INITIAL_CAPITAL = 1000000.00
STRICT_MAX_OPEN_POSITIONS = 15
EQUITY_CAP_FOR_RISK_CALC_MULTIPLE = 25

# --- GLOBAL MARKET REGIME FILTER ---
USE_TREND_AGNOSTIC_MODE = True
TREND_FILTER_SMA_PERIOD = 100

# --- Trading Session ---
EOD_TIME = "15:15"
INDIA_TZ = timezone('Asia/Kolkata')

# --- File Paths ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LONGS_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_longs_data_with_signals_and_atr.parquet")
SHORTS_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "tfl_shorts_data_with_signals_and_atr.parquet")
REGIME_DATA_PATH = os.path.join(ROOT_DIR, "data", "strategy_specific_data", "market_regime_data.parquet")
LOGS_BASE_DIR = os.path.join(ROOT_DIR, "backtest_logs")

# ==============================================================================
# --- STRATEGY PROFILES ---
# ==============================================================================
STRATEGY_PROFILES = {
    "LONGS_ONLY": {
        "direction": "LONG",
        "use_breadth_filter": False,
        "breadth_threshold_pct": 60.0,
        "use_volatility_filter": False,
        "vix_threshold": 17.0,
        "entry_type_to_use": 'fast',
        "risk_per_trade_pct": 0.01,
        "max_total_risk_pct": 0.05,
        "max_capital_per_trade_pct": 0.25,
        "slippage_pct": 0.05 / 100,
        "transaction_cost_pct": 0.03 / 100,
        "risk_reward_ratio": 10.0,
        "atr_ts_period": 14,
        "atr_ts_multiplier": 4.0,
        "breakeven_trigger_r": 1.0,
        "breakeven_profit_r": 0.1,
        "aggressive_ts_trigger_r": 5.0,
        "aggressive_ts_multiplier": 1.0,
        "exit_on_eod": True,
        "allow_afternoon_positional": False,
        "afternoon_entry_threshold": "14:00",
        "avoid_open_close_entries": True
    },
    "SHORTS_ONLY": {
        "direction": "SHORT",
        "use_breadth_filter": False,
        "breadth_threshold_pct": 60.0,
        "use_volatility_filter": False,
        "vix_threshold": 17.0,
        "entry_type_to_use": 'fast',
        "risk_per_trade_pct": 0.01,
        "max_total_risk_pct": 0.05,
        "max_capital_per_trade_pct": 0.25,
        "slippage_pct": 0.05 / 100,
        "transaction_cost_pct": 0.03 / 100,
        "risk_reward_ratio": 10.0,
        "atr_ts_period": 14,
        "atr_ts_multiplier": 3.0,
        "breakeven_trigger_r": 1.0,
        "breakeven_profit_r": 0.1,
        "aggressive_ts_trigger_r": 3.0,
        "aggressive_ts_multiplier": 1.0,
        "exit_on_eod": True,
        "allow_afternoon_positional": False,
        "afternoon_entry_threshold": "14:00",
        "avoid_open_close_entries": True
    }
}

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================
def setup_logging_and_dirs(log_base_dir, strategy_name):
    strategy_name_with_version = f"{strategy_name}_v3.1"
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_base_dir, strategy_name_with_version, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def log_configs(log_dir):
    with open(os.path.join(log_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"--- Strategy: {STRATEGY_NAME} (v3.1) ---\n")
        f.write(f"--- Initial Capital: ₹{INITIAL_CAPITAL:,.2f} ---\n")
        f.write(f"--- Equity Cap for Risk: {EQUITY_CAP_FOR_RISK_CALC_MULTIPLE}x ---\n")
        f.write(f"--- TREND AGNOSTIC MODE: {USE_TREND_AGNOSTIC_MODE} ---\n")
        f.write(f"--- GLOBAL TREND FILTER SMA: {TREND_FILTER_SMA_PERIOD} ---\n")
        f.write("\n--- LONG STRATEGY PROFILE ---\n")
        for key, value in STRATEGY_PROFILES["LONGS_ONLY"].items(): f.write(f"- {key}: {value}\n")
        f.write("\n--- SHORT STRATEGY PROFILE ---\n")
        for key, value in STRATEGY_PROFILES["SHORTS_ONLY"].items(): f.write(f"- {key}: {value}\n")
        f.write("-" * 30 + "\n\n")

def calculate_and_log_metrics(log_dir, closed_trades_df, equity_curve_df):
    if closed_trades_df.empty:
        summary = "No trades were executed."
        with open(os.path.join(log_dir, 'summary.txt'), 'a', encoding='utf-8') as f: f.write(summary)
        return

    # --- Overall Performance ---
    total_trades = len(closed_trades_df)
    wins = closed_trades_df[closed_trades_df['pnl'] > 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    net_pnl = closed_trades_df['pnl'].sum()
    gross_profit = wins['pnl'].sum()
    gross_loss = closed_trades_df[closed_trades_df['pnl'] <= 0]['pnl'].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
    
    end_equity = equity_curve_df['equity'].iloc[-1]
    start_equity = INITIAL_CAPITAL
    num_years = (equity_curve_df.index[-1] - equity_curve_df.index[0]).days / 365.25
    cagr = ((end_equity / start_equity) ** (1 / num_years) - 1) * 100 if num_years > 0 and start_equity > 0 else 0

    daily_returns = equity_curve_df['equity'].resample('D').last().pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    max_drawdown = (equity_curve_df['equity'] / equity_curve_df['equity'].cummax() - 1).min() * 100
    
    summary = (
        f"--- Overall Performance Metrics ---\n"
        f"- Net PnL:                  ₹{net_pnl:,.2f}\n"
        f"- Total Trades:             {total_trades}\n"
        f"- Win Rate:                 {win_rate:.2f}%\n"
        f"- Profit Factor:            {profit_factor:.2f}\n"
        f"- CAGR:                     {cagr:.2f}%\n"
        f"- Sharpe Ratio (Annualized): {sharpe_ratio:.2f}\n"
        f"- Max Drawdown:             {max_drawdown:.2f}%\n"
    )

    # --- ENHANCED v3.1: Strategy Breakdown with Profit Factor ---
    long_trades = closed_trades_df[closed_trades_df['direction'] == 'LONG']
    short_trades = closed_trades_df[closed_trades_df['direction'] == 'SHORT']

    summary += "\n--- Strategy Breakdown ---\n"
    if not long_trades.empty:
        long_wins = long_trades[long_trades['pnl'] > 0]
        long_gross_profit = long_wins['pnl'].sum()
        long_gross_loss = long_trades[long_trades['pnl'] <= 0]['pnl'].sum()
        long_profit_factor = abs(long_gross_profit / long_gross_loss) if long_gross_loss != 0 else float('inf')
        summary += (
            f"- Long Trades:              {len(long_trades)}\n"
            f"- Long Win Rate:            {(len(long_wins) / len(long_trades)) * 100:.2f}%\n"
            f"- Long Net PnL:             ₹{long_trades['pnl'].sum():,.2f}\n"
            f"- Long Profit Factor:       {long_profit_factor:.2f}\n"
        )
    if not short_trades.empty:
        short_wins = short_trades[short_trades['pnl'] > 0]
        short_gross_profit = short_wins['pnl'].sum()
        short_gross_loss = short_trades[short_trades['pnl'] <= 0]['pnl'].sum()
        short_profit_factor = abs(short_gross_profit / short_gross_loss) if short_gross_loss != 0 else float('inf')
        summary += (
            f"- Short Trades:             {len(short_trades)}\n"
            f"- Short Win Rate:           {(len(short_wins) / len(short_trades)) * 100:.2f}%\n"
            f"- Short Net PnL:            ₹{short_trades['pnl'].sum():,.2f}\n"
            f"- Short Profit Factor:      {short_profit_factor:.2f}\n"
        )

    # --- ENHANCED v3.1: Year-over-Year Performance with readable format ---
    if not closed_trades_df.empty:
        closed_trades_df['year'] = pd.to_datetime(closed_trades_df['entry_time']).dt.year
        yoy_summary = closed_trades_df.groupby('year')['pnl'].agg(['sum', 'count']).rename(columns={'sum': 'net_pnl', 'count': 'trade_count'})
        yoy_summary['net_pnl_formatted'] = yoy_summary['net_pnl'].apply(lambda x: f"₹{x:,.2f}")
        
        summary += "\n--- Year-over-Year Performance ---\n"
        summary += yoy_summary[['net_pnl_formatted', 'trade_count']].to_string()

    with open(os.path.join(log_dir, 'summary.txt'), 'a', encoding='utf-8') as f:
        f.write(summary)


def update_progress(progress, total, current_date, equity, open_positions, bias):
    bar_length, percent = 40, float(progress) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    equity_str = f"Equity: ₹{equity:,.2f}"
    open_pos_str = f"Open: {len(open_positions)}"
    bias_str = f"Bias: {bias}"
    sys.stdout.write(f"\rDate: {current_date.strftime('%Y-%m-%d')} | {equity_str} | {open_pos_str} | {bias_str} | Progress: [{arrow + spaces}] {percent*100:.1f}%")
    sys.stdout.flush()

# ==============================================================================
# --- MAIN SIMULATION ENGINE ---
# ==============================================================================
def main():
    log_dir = setup_logging_and_dirs(LOGS_BASE_DIR, STRATEGY_NAME)
    log_configs(log_dir)
    print(f"Starting UNIFIED backtest: {STRATEGY_NAME} v3.1. Log files will be saved in:\n{log_dir}\n")

    try:
        longs_df = pd.read_parquet(LONGS_DATA_PATH)
        shorts_df = pd.read_parquet(SHORTS_DATA_PATH)
        regime_df = pd.read_parquet(REGIME_DATA_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: A required data file was not found. Details: {e}"); return
    
    longs_df['direction'] = 'LONG'
    shorts_df['direction'] = 'SHORT'
    longs_df.reset_index(inplace=True)
    shorts_df.reset_index(inplace=True)
    master_df = pd.concat([longs_df, shorts_df], ignore_index=True)
    master_df.drop_duplicates(subset=['datetime', 'symbol', 'direction'], keep='first', inplace=True)
    master_df.set_index('datetime', inplace=True)
    
    master_df.sort_values(by=['symbol', 'datetime'], inplace=True)
    master_df['pattern_low_for_sl'] = master_df.groupby('symbol')['pattern_low'].shift(1)
    master_df['pattern_high_for_sl'] = master_df.groupby('symbol')['pattern_high'].shift(1)
    
    master_df.sort_index(inplace=True)
    merged_df = pd.merge_asof(master_df, regime_df.sort_index(), left_index=True, right_index=True, direction='backward')
    
    start_ts = INDIA_TZ.localize(pd.to_datetime(START_DATE))
    end_ts = INDIA_TZ.localize(pd.to_datetime(END_DATE))
    merged_df = merged_df[(merged_df.index >= start_ts) & (merged_df.index <= end_ts)]
    
    timestamps = sorted(merged_df.index.unique())
    if not timestamps: print("No data in date range."); return
    
    cash, equity = INITIAL_CAPITAL, INITIAL_CAPITAL
    open_positions, closed_trades_log, rejected_trades_log = [], [], []
    equity_curve = [{'datetime': timestamps[0], 'equity': INITIAL_CAPITAL}]
    last_known_prices = {}
    trading_bias_for_the_day = "NO_TRADES"
    
    equity_cap = INITIAL_CAPITAL * EQUITY_CAP_FOR_RISK_CALC_MULTIPLE if EQUITY_CAP_FOR_RISK_CALC_MULTIPLE > 0 else float('inf')
    
    daily_capital_log = []
    idle_capital_today = 0

    total_timestamps = len(timestamps)
    for i, ts in enumerate(timestamps):
        current_data_slice = merged_df.loc[ts]
        if isinstance(current_data_slice, pd.Series):
             current_data_slice = current_data_slice.to_frame().T
        
        new_prices = current_data_slice.set_index('symbol').to_dict('index')
        last_known_prices.update(new_prices)
        
        if i == 0 or ts.date() != timestamps[i-1].date():
            if i > 0:
                prev_day_ts = timestamps[i-1]
                market_value_longs = sum(p['quantity'] * last_known_prices.get(p['symbol'], {}).get('close', p['entry_price']) for p in open_positions if p['direction'] == 'LONG')
                market_value_shorts = sum(p['quantity'] * last_known_prices.get(p['symbol'], {}).get('close', p['entry_price']) for p in open_positions if p['direction'] == 'SHORT')
                deployed_capital = market_value_longs + sum(p.get('initial_proceeds', 0) for p in open_positions if p['direction'] == 'SHORT')
                daily_capital_log.append({
                    'date': prev_day_ts.date(), 'end_of_day_equity': equity,
                    'capital_deployed': deployed_capital, 'idle_capital_rejected': idle_capital_today
                })
                idle_capital_today = 0
            
            if USE_TREND_AGNOSTIC_MODE:
                trading_bias_for_the_day = "ALL_TRADES"
            else:
                today_regime = current_data_slice.iloc[0]
                sma_col_name = f'is_nifty_above_sma_{TREND_FILTER_SMA_PERIOD}'
                if today_regime.get(sma_col_name, False): 
                    trading_bias_for_the_day = "LONGS_ONLY"
                else:
                    trading_bias_for_the_day = "SHORTS_ONLY"
            update_progress(i + 1, total_timestamps, ts, equity, open_positions, trading_bias_for_the_day)

        positions_to_close = []
        for trade in open_positions:
            candle = last_known_prices.get(trade['symbol'])
            if candle is None: continue
            
            profile = STRATEGY_PROFILES[trade['direction'] + "S_ONLY"]
            exit_reason, exit_price = None, None
            
            if trade['direction'] == 'LONG':
                current_profit_r = (candle['high'] - trade['entry_price']) / trade['initial_risk_per_share'] if trade['initial_risk_per_share'] > 0 else 0
                if profile['breakeven_trigger_r'] and not trade['be_activated'] and current_profit_r >= profile['breakeven_trigger_r']:
                    trade['sl'] = trade['entry_price'] + (trade['initial_risk_per_share'] * profile['breakeven_profit_r']); trade['be_activated'] = True
                if profile['aggressive_ts_trigger_r'] and current_profit_r >= profile['aggressive_ts_trigger_r']:
                    trade['current_ts_multiplier'] = profile['aggressive_ts_multiplier']
                atr_col_name = f'atr_{profile["atr_ts_period"]}'
                if profile['atr_ts_period'] and pd.notna(candle[atr_col_name]):
                    trade['sl'] = max(trade['sl'], candle['high'] - (candle[atr_col_name] * trade['current_ts_multiplier']))
                
                if candle['low'] <= trade['sl']: exit_reason, exit_price = 'SL_HIT', trade['sl']
                elif candle['high'] >= trade['tp']: exit_reason, exit_price = 'TP_HIT', trade['tp']
            else: # SHORT
                current_profit_r = (trade['entry_price'] - candle['low']) / trade['initial_risk_per_share'] if trade['initial_risk_per_share'] > 0 else 0
                if profile['breakeven_trigger_r'] and not trade['be_activated'] and current_profit_r >= profile['breakeven_trigger_r']:
                    trade['sl'] = trade['entry_price'] - (trade['initial_risk_per_share'] * profile['breakeven_profit_r']); trade['be_activated'] = True
                if profile['aggressive_ts_trigger_r'] and current_profit_r >= profile['aggressive_ts_trigger_r']:
                    trade['current_ts_multiplier'] = profile['aggressive_ts_multiplier']
                atr_col_name = f'atr_{profile["atr_ts_period"]}'
                if profile['atr_ts_period'] and pd.notna(candle.get(atr_col_name)):
                    trade['sl'] = min(trade['sl'], candle['low'] + (candle[atr_col_name] * trade['current_ts_multiplier']))

                if candle['high'] >= trade['sl']: exit_reason, exit_price = 'SL_HIT', trade['sl']
                elif candle['low'] <= trade['tp']: exit_reason, exit_price = 'TP_HIT', trade['tp']

            if profile['exit_on_eod'] and ts.strftime('%H:%M') == EOD_TIME and not exit_reason:
                 if profile['allow_afternoon_positional'] and trade.get('is_afternoon_entry', False): continue 
                 exit_reason, exit_price = 'EOD_EXIT', candle['close']
            
            if exit_reason:
                if trade['direction'] == 'LONG':
                    net_proceeds = (trade['quantity'] * exit_price) * (1 - profile['transaction_cost_pct'])
                    cash += net_proceeds
                    pnl = net_proceeds - trade['initial_cost_with_fees']
                else: # SHORT
                    cost_to_cover = (trade['quantity'] * exit_price) * (1 + profile['transaction_cost_pct'])
                    cash -= cost_to_cover
                    pnl = trade['initial_proceeds'] - cost_to_cover
                
                trade.update({'exit_time': ts, 'exit_price': exit_price, 'exit_reason': exit_reason, 'pnl': pnl})
                closed_trades_log.append(trade)
                positions_to_close.append(trade)

        open_positions = [p for p in open_positions if p not in positions_to_close]
        total_current_risk_value = sum(trade.get('initial_risk_value', 0) for trade in open_positions)
        
        if trading_bias_for_the_day != "NO_TRADES":
            
            if trading_bias_for_the_day == "ALL_TRADES":
                potential_trades_df = current_data_slice
            else:
                profile = STRATEGY_PROFILES[trading_bias_for_the_day]
                potential_trades_df = current_data_slice[current_data_slice['direction'] == profile['direction']]
            
            if not potential_trades_df.empty:
                active_symbols = [p['symbol'] for p in open_positions]
                
                for _, signal in potential_trades_df.iterrows():
                    direction_key = signal['direction'] + "S_ONLY"
                    profile = STRATEGY_PROFILES[direction_key]
                    
                    equity_for_risk_calc = min(equity, equity_cap)
                    
                    current_time_str = ts.strftime('%H:%M')
                    if profile['avoid_open_close_entries'] and (current_time_str == '09:15' or current_time_str == '15:15'):
                        continue
                    
                    if signal['symbol'] in active_symbols: continue

                    entry_signal_col = f"is_{profile['entry_type_to_use']}_entry"
                    if not signal.get(entry_signal_col, False): continue
                        
                    available_risk_budget = (equity_for_risk_calc * profile['max_total_risk_pct']) - total_current_risk_value
                    desired_risk_amount = equity_for_risk_calc * profile['risk_per_trade_pct']
                    risk_amount = min(desired_risk_amount, available_risk_budget)
                    if risk_amount <= 0:
                        rejected_trades_log.append({'timestamp': ts, 'symbol': signal['symbol'], 'reason': 'MAX_TOTAL_RISK_EXCEEDED'}); continue
                    
                    price_col = f"{profile['entry_type_to_use']}_entry_price"
                    entry_price = signal[price_col]

                    if profile['direction'] == 'LONG':
                        initial_sl = signal['pattern_low_for_sl']
                        risk_per_share = entry_price - initial_sl
                    else: # SHORT
                        initial_sl = signal['pattern_high_for_sl']
                        risk_per_share = initial_sl - entry_price

                    if pd.isna(initial_sl) or risk_per_share <= 0: continue
                    
                    quantity_by_risk = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                    capital_for_trade = equity_for_risk_calc * profile['max_capital_per_trade_pct']
                    quantity_by_capital = int(capital_for_trade / entry_price) if entry_price > 0 else 0
                    quantity = min(quantity_by_risk, quantity_by_capital)
                    
                    if len(open_positions) >= STRICT_MAX_OPEN_POSITIONS:
                        rejected_trades_log.append({'timestamp': ts, 'symbol': signal['symbol'], 'reason': 'MAX_POSITIONS_REACHED'})
                        if quantity > 0: idle_capital_today += quantity * entry_price
                        continue

                    actual_risk_value = quantity * risk_per_share
                    
                    if profile['direction'] == 'LONG':
                        cost = quantity * entry_price * (1 + profile['transaction_cost_pct'])
                        if quantity > 0 and cash >= cost:
                            cash -= cost
                            new_trade = { 'symbol': signal['symbol'], 'direction': 'LONG', 'entry_time': ts, 'entry_price': entry_price, 'quantity': quantity, 'sl': initial_sl, 'tp': entry_price + (risk_per_share * profile['risk_reward_ratio']), 'initial_risk_per_share': risk_per_share, 'initial_risk_value': actual_risk_value, 'initial_sl': initial_sl, 'be_activated': False, 'current_ts_multiplier': profile['atr_ts_multiplier'], 'is_afternoon_entry': ts.strftime('%H:%M') >= profile['afternoon_entry_threshold'], 'initial_cost_with_fees': cost }
                            open_positions.append(new_trade)
                            total_current_risk_value += actual_risk_value
                    else: # SHORT
                        if quantity > 0 and cash >= actual_risk_value:
                            initial_proceeds = (quantity * entry_price) * (1 - profile['transaction_cost_pct'])
                            cash += initial_proceeds
                            new_trade = { 'symbol': signal['symbol'], 'direction': 'SHORT', 'entry_time': ts, 'entry_price': entry_price, 'quantity': quantity, 'sl': initial_sl, 'tp': entry_price - (risk_per_share * profile['risk_reward_ratio']), 'initial_risk_per_share': risk_per_share, 'initial_risk_value': actual_risk_value, 'initial_sl': initial_sl, 'be_activated': False, 'current_ts_multiplier': profile['atr_ts_multiplier'], 'is_afternoon_entry': ts.strftime('%H:%M') >= profile['afternoon_entry_threshold'], 'initial_proceeds': initial_proceeds }
                            open_positions.append(new_trade)
                            total_current_risk_value += actual_risk_value
        
        market_value_longs = sum(p['quantity'] * last_known_prices.get(p['symbol'], {}).get('close', 0) for p in open_positions if p.get('direction') == 'LONG')
        market_value_shorts = sum(p['quantity'] * last_known_prices.get(p['symbol'], {}).get('close', 0) for p in open_positions if p.get('direction') == 'SHORT')
        equity = cash + market_value_longs - market_value_shorts
        equity_curve.append({'datetime': ts, 'equity': equity})
        
    update_progress(total_timestamps, total_timestamps, timestamps[-1], equity, open_positions, trading_bias_for_the_day)
    print("\n\n--- Backtest Simulation Finished ---")

    closed_df = pd.DataFrame(closed_trades_log)
    rejected_df = pd.DataFrame(rejected_trades_log)
    capital_df = pd.DataFrame(daily_capital_log)
    equity_df = pd.DataFrame(equity_curve).set_index('datetime')
    
    if not capital_df.empty: capital_df.to_csv(os.path.join(log_dir, 'capital_utilization_log.csv'), index=False)
    if not closed_df.empty: closed_df.to_csv(os.path.join(log_dir, 'trade_log.csv'), index=False)
    if not rejected_df.empty: rejected_df.to_csv(os.path.join(log_dir, 'rejected_trades_log.csv'), index=False)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    equity_df['equity'].plot(figsize=(12, 8), title='Portfolio Equity Curve', color='blue')
    plt.ylabel('Equity (₹)'); plt.xlabel('Date'); plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'equity_curve.png'))
    
    calculate_and_log_metrics(log_dir, closed_df, equity_df)
    print(f"Results and logs have been saved to: {log_dir}")

if __name__ == "__main__":
    main()

