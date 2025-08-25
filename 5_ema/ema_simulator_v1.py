import pandas as pd
import numpy as np
import sys
import os
from datetime import time, datetime
import json

# ==============================================================================
# --- BACKTEST CONFIGURATION ---
# ==============================================================================
# This dictionary holds all the parameters for the backtest.
# You can easily modify these values to test different scenarios.
CONFIG = {
    # --- Portfolio & Strategy Details ---
    "strategy_name": "5-EMA-Manny",
    "initial_portfolio_value_inr": 500000.0,
    # --- Universe Configuration ---
    "symbols": ["NIFTYBANK-INDEX"], 
    "lot_sizes": {
        "NIFTYBANK-INDEX": 35, # Corrected to current lot size
        "NIFTY50-INDEX": 25,  # Corrected to current lot size
        # Add other symbols and their lot sizes here
    },
    "timeframe": "15min",

    # --- Data File Configuration ---
    "data_folder": "../data/universal_processed/15min",
    "data_file_name": "{symbol}_{timeframe}_with_indicators.csv",

    # --- Date Range for Backtest ---
    "start_date": "2022-01-01",
    "end_date": "2023-12-31",

    # --- Trading Session Time ---
    "trade_window_start": time(9, 45),
    "trade_window_end": time(15, 15),
    "last_entry_time": time(15, 0),

    # --- Strategy Parameters ---
    "ema_period": 5,
    "trade_mode": "short_only",
    "alert_candle_color": "required",
    "trade_hold_duration_candles_long": 10,
    "trade_hold_duration_candles_short": 20,

    # --- Entry Filters ---
    "rsi_filter_enabled": False,
    "rsi_period": 14,
    "rsi_overbought": 60.0,
    "rsi_oversold": 40.0,
    "volume_filter_enabled": False,
    "volume_ratio_min": 1.5,
    "body_ratio_filter_enabled": False,
    "body_ratio_min": 0.6,

    # --- Risk & Order Management ---
    "sl_points_default": 40.0,
    "entry_buffer_pips": 2.0,
    "slippage_pips": 1.5,
    "transaction_costs_per_trade_points": 12.0, # Points to deduct for taxes/charges
    "use_breakeven_sl": True, # Set to True to enable aggressive break-even
    "breakeven_profit_points": 12.0, # Points above/below entry to set SL once in profit

    # --- Trailing Stop Loss Parameters ---
    "trailing_sl_type": "atr",
    "trailing_sl_activation_delay": 5,
    "trailing_sl_atr_period": 14,
    "trailing_sl_atr_multiplier": 2.75,

    # --- Reporting ---
    "log_folder": "backtest_logs",
    "risk_free_rate_annual": 0.07
}

# ==============================================================================
# --- BACKTESTING ENGINE ---
# ==============================================================================

class MeanReversionBacktester:
    """
    A class to backtest the '5-EMA-Manny' mean-reversion strategy for a single symbol.
    """
    def __init__(self, config, symbol):
        self.config = config
        self.symbol = symbol
        self.lot_size = config['lot_sizes'].get(symbol, 1) # Default to 1 if not found
        self.df = None
        self.trades = []
        self.in_position = False
        self.current_trade = {}

    def get_trades_df(self):
        return pd.DataFrame(self.trades)

    def _calculate_atr(self):
        if self.config.get("trailing_sl_type") == 'atr':
            atr_period = self.config.get("trailing_sl_atr_period", 14)
            high_low = self.df['high'] - self.df['low']
            high_close = np.abs(self.df['high'] - self.df['close'].shift())
            low_close = np.abs(self.df['low'] - self.df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            self.df[f'atr_{atr_period}'] = true_range.rolling(atr_period).mean()

    def _load_data(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(script_dir, self.config["data_folder"])
            file_path = os.path.normpath(os.path.join(
                base_path,
                self.config["data_file_name"].format(symbol=self.symbol, timeframe=self.config["timeframe"])
            ))
            print(f"Loading data for {self.symbol} from: {file_path}\n")
            self.df = pd.read_csv(file_path, index_col='datetime', parse_dates=True).sort_index()
            self._calculate_atr()

            start_date = self.config.get("start_date")
            end_date = self.config.get("end_date")
            if start_date: self.df = self.df[self.df.index >= pd.to_datetime(start_date)]
            if end_date: self.df = self.df[self.df.index <= pd.to_datetime(end_date)]

            if self.df.empty:
                print(f"Warning: No data available for {self.symbol} in the specified date range.")
                return False
            if f'ema_{self.config["ema_period"]}' not in self.df.columns:
                raise ValueError(f"Required EMA column not found for {self.symbol}.")
        except FileNotFoundError:
            print(f"Warning: Data file not found for {self.symbol} at '{file_path}'. Skipping.")
            return False
        except Exception as e:
            print(f"An error occurred loading data for {self.symbol}: {e}")
            return False
        return True

    def _log_trade(self, exit_reason):
        # Calculate gross PnL in points
        gross_pnl_points = self.current_trade['exit_price'] - self.current_trade['entry_price']
        if self.current_trade['type'] == 'SHORT':
            gross_pnl_points *= -1
        
        # Deduct transaction costs to get net PnL
        costs_in_points = self.config.get("transaction_costs_per_trade_points", 0)
        net_pnl_points = gross_pnl_points - costs_in_points

        self.trades.append({
            "Strategy": self.config['strategy_name'],
            "Symbol": self.symbol,
            "EntryTime": self.current_trade['entry_time'],
            "EntryPrice": self.current_trade['entry_price'],
            "TradeType": self.current_trade['type'],
            "ExitTime": self.current_trade['exit_time'],
            "ExitPrice": self.current_trade['exit_price'],
            "SL": self.current_trade['sl'],
            "PnL_Points": round(net_pnl_points, 2), # Log net points
            "PnL_INR": round(net_pnl_points * self.lot_size, 2), # Log net INR
            "ExitReason": exit_reason,
        })
        self.in_position = False
        self.current_trade = {}

    def run_backtest(self):
        if not self._load_data(): return
        total_candles = len(self.df)
        print(f"Starting backtest simulation for {self.symbol}...")

        for i in range(2, total_candles):
            self._update_progress(i, total_candles)
            current_candle = self.df.iloc[i]
            t_minus_1_candle = self.df.iloc[i-1]
            t_minus_2_candle = self.df.iloc[i-2]
            current_time = current_candle.name.time()
            ema_col = f'ema_{self.config["ema_period"]}'

            if self.in_position:
                exit_price, exit_reason = 0, None
                
                # --- Aggressive Break-even Logic ---
                if self.config.get("use_breakeven_sl") and not self.current_trade.get('breakeven_set', False):
                    profit_target = self.config.get("breakeven_profit_points", 12.0)
                    if self.current_trade['type'] == 'LONG':
                        # If price is profitable, move SL to break-even + profit target
                        if current_candle['close'] > self.current_trade['entry_price']:
                            new_sl = self.current_trade['entry_price'] + profit_target
                            self.current_trade['sl'] = max(self.current_trade['sl'], new_sl)
                            self.current_trade['breakeven_set'] = True
                    else: # SHORT
                        # If price is profitable, move SL to break-even - profit target
                        if current_candle['close'] < self.current_trade['entry_price']:
                            new_sl = self.current_trade['entry_price'] - profit_target
                            self.current_trade['sl'] = min(self.current_trade['sl'], new_sl)
                            self.current_trade['breakeven_set'] = True

                # --- Trailing Stop Logic ---
                trailing_sl_type = self.config.get("trailing_sl_type")
                candles_held = i - self.current_trade['entry_index']
                activation_delay = self.config.get("trailing_sl_activation_delay", 1)

                if trailing_sl_type and candles_held >= activation_delay:
                    if trailing_sl_type == 'atr':
                        atr_value = current_candle.get(f'atr_{self.config.get("trailing_sl_atr_period", 14)}')
                        if not pd.isna(atr_value):
                            atr_offset = atr_value * self.config.get("trailing_sl_atr_multiplier", 2.0)
                            if self.current_trade['type'] == 'LONG':
                                self.current_trade['sl'] = max(self.current_trade['sl'], current_candle['high'] - atr_offset)
                            else:
                                self.current_trade['sl'] = min(self.current_trade['sl'], current_candle['low'] + atr_offset)
                    elif trailing_sl_type == 'candle':
                        if self.current_trade['type'] == 'LONG' and current_candle['close'] > current_candle['open']:
                            self.current_trade['sl'] = max(self.current_trade['sl'], current_candle['low'])
                        elif self.current_trade['type'] == 'SHORT' and current_candle['open'] > current_candle['close']:
                            self.current_trade['sl'] = min(self.current_trade['sl'], current_candle['high'])

                # --- Exit Condition Checks ---
                if self.current_trade['type'] == 'LONG' and current_candle['low'] <= self.current_trade['sl']:
                    exit_price, exit_reason = self.current_trade['sl'], "Trailing SL Hit" if trailing_sl_type else "SL Hit"
                elif self.current_trade['type'] == 'SHORT' and current_candle['high'] >= self.current_trade['sl']:
                    exit_price, exit_reason = self.current_trade['sl'], "Trailing SL Hit" if trailing_sl_type else "SL Hit"
                elif not exit_reason and current_candle.name >= self.current_trade['exit_time_target']:
                    exit_price, exit_reason = current_candle['close'], "Time Exit"
                elif not exit_reason and current_time >= self.config['trade_window_end']:
                    exit_price, exit_reason = current_candle['close'], "End of Day"

                if exit_reason:
                    slippage = self.config['slippage_pips']
                    final_exit_price = exit_price - slippage if self.current_trade['type'] == 'LONG' else exit_price + slippage
                    self.current_trade.update({'exit_price': final_exit_price, 'exit_time': current_candle.name})
                    self._log_trade(exit_reason)
                    continue

            if self.config['trade_window_start'] <= current_time < self.config['last_entry_time'] and not self.in_position:
                trade_params = self._check_entry_signals(current_candle, t_minus_1_candle, t_minus_2_candle, ema_col, i)
                if trade_params:
                    self.in_position = True
                    self.current_trade = trade_params
                    continue
        print(f"\nBacktest simulation finished for {self.symbol}.")

    def _check_entry_signals(self, current_candle, t_minus_1, t_minus_2, ema_col, index):
        # Check SHORT signals
        if self.config['trade_mode'] in ['short_only', 'both']:
            is_valid_color = (self.config['alert_candle_color'] != 'required') or (t_minus_1['open'] > t_minus_1['close'])
            if is_valid_color and t_minus_1['low'] > t_minus_1[ema_col] and self._filters_passed('SHORT', t_minus_1):
                trigger_price = t_minus_1['low'] - self.config['entry_buffer_pips']
                if current_candle['low'] < trigger_price:
                    entry_price = trigger_price - self.config['slippage_pips']
                    sl = max(max(t_minus_1['high'], t_minus_2['high']), entry_price + self.config['sl_points_default'])
                    hold_duration = self.config['trade_hold_duration_candles_short']
                    return {'type': 'SHORT', 'entry_time': current_candle.name, 'entry_price': entry_price, 'sl': sl,
                            'exit_time_target': self.df.index[min(index + hold_duration - 1, len(self.df)-1)], 'entry_index': index, 'breakeven_set': False}
        # Check LONG signals
        if self.config['trade_mode'] in ['long_only', 'both']:
            is_valid_color = (self.config['alert_candle_color'] != 'required') or (t_minus_1['close'] > t_minus_1['open'])
            if is_valid_color and t_minus_1['high'] < t_minus_1[ema_col] and self._filters_passed('LONG', t_minus_1):
                trigger_price = t_minus_1['high'] + self.config['entry_buffer_pips']
                if current_candle['high'] > trigger_price:
                    entry_price = trigger_price + self.config['slippage_pips']
                    sl = min(min(t_minus_1['low'], t_minus_2['low']), entry_price - self.config['sl_points_default'])
                    hold_duration = self.config['trade_hold_duration_candles_long']
                    return {'type': 'LONG', 'entry_time': current_candle.name, 'entry_price': entry_price, 'sl': sl,
                            'exit_time_target': self.df.index[min(index + hold_duration - 1, len(self.df)-1)], 'entry_index': index, 'breakeven_set': False}
        return None

    def _filters_passed(self, trade_type, candle):
        if self.config['rsi_filter_enabled']:
            rsi_val = candle[f"rsi_{self.config['rsi_period']}"]
            if trade_type == 'SHORT' and rsi_val <= self.config['rsi_overbought']: return False
            if trade_type == 'LONG' and rsi_val >= self.config['rsi_oversold']: return False
        if self.config['volume_filter_enabled'] and candle['volume_ratio'] <= self.config['volume_ratio_min']: return False
        if self.config['body_ratio_filter_enabled'] and candle['body_ratio'] <= self.config['body_ratio_min']: return False
        return True

    def _update_progress(self, current, total):
        progress = current / total
        bar = '#' * int(40 * progress)
        sys.stdout.write(f"\r > {self.symbol}: Processing: [{bar:<40}] {current}/{total} ({progress:.1%})")
        sys.stdout.flush()

def generate_performance_report(log_df, config, report_level="Portfolio"):
    if log_df.empty:
        print(f"\n--- No Trades to Analyze for {report_level} ---")
        return

    total_trades = len(log_df)
    wins = log_df[log_df['PnL_INR'] > 0]
    win_rate = len(wins) / total_trades * 100
    total_pnl_inr = log_df['PnL_INR'].sum()
    gross_profit = wins['PnL_INR'].sum()
    gross_loss = abs(log_df[log_df['PnL_INR'] <= 0]['PnL_INR'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    log_df = log_df.sort_values(by='ExitTime')
    initial_capital = config['initial_portfolio_value_inr']
    log_df['Equity'] = initial_capital + log_df['PnL_INR'].cumsum()
    log_df['Peak_Equity'] = log_df['Equity'].cummax()
    log_df['Drawdown_INR'] = log_df['Peak_Equity'] - log_df['Equity']
    log_df['Drawdown_%'] = (log_df['Drawdown_INR'] / log_df['Peak_Equity']) * 100
    max_dd_inr = log_df['Drawdown_INR'].max()
    max_dd_pct = log_df['Drawdown_%'].max()

    # --- Advanced Metrics Calculation ---
    # CAGR
    ending_equity = log_df['Equity'].iloc[-1]
    start_date = log_df['EntryTime'].iloc[0]
    end_date = log_df['ExitTime'].iloc[-1]
    years = (end_date - start_date).days / 365.25
    cagr = ((ending_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sortino and Calmar Ratios
    daily_pnl = log_df.set_index('ExitTime')['PnL_INR'].resample('D').sum()
    daily_returns = daily_pnl / initial_capital
    annualized_return = daily_returns.mean() * 252
    
    downside_returns = daily_returns[daily_returns < 0]
    downside_std_dev = downside_returns.std() * np.sqrt(252)
    
    sortino_ratio = annualized_return / downside_std_dev if downside_std_dev > 0 else 0
    calmar_ratio = (annualized_return * 100) / max_dd_pct if max_dd_pct > 0 else 0

    header = f"--- {report_level} Performance Summary ---"
    print(f"\n{header}")
    if report_level != "Portfolio": print(f" Symbol:              {log_df['Symbol'].iloc[0]}")
    print("-" * len(header))
    print(f" Total Trades:        {total_trades}")
    print(f" Win Rate:            {win_rate:.2f}%")
    print(f" Profit Factor:       {profit_factor:.2f}")
    print(f" CAGR:                {cagr:.2f}%")
    print(f" Sortino Ratio:       {sortino_ratio:.2f}")
    print(f" Calmar Ratio:        {calmar_ratio:.2f}")
    print(f" Max Drawdown (INR):  {max_dd_inr:,.2f}")
    print(f" Max Drawdown (%):    {max_dd_pct:.2f}%")
    print("------------------------------------")
    print(f" Total Net PnL (INR): {total_pnl_inr:,.2f}")
    print("--- End of Report ---\n")

def save_config_summary(config, file_path):
    with open(file_path, 'w') as f:
        f.write("--- Backtest Configuration Summary ---\n")
        json.dump({k: v.strftime('%H:%M:%S') if isinstance(v, time) else v for k, v in config.items()}, f, indent=4)
    print(f"Configuration summary saved to: {file_path}")

if __name__ == '__main__':
    all_trades_list = []
    for symbol in CONFIG['symbols']:
        print(f"\n{'='*60}\nProcessing Symbol: {symbol}\n{'='*60}")
        backtester = MeanReversionBacktester(CONFIG, symbol)
        backtester.run_backtest()
        symbol_trades_df = backtester.get_trades_df()
        if not symbol_trades_df.empty:
            all_trades_list.append(symbol_trades_df)
            generate_performance_report(symbol_trades_df, CONFIG, report_level="Individual Symbol")

    if all_trades_list:
        portfolio_df = pd.concat(all_trades_list, ignore_index=True)
        generate_performance_report(portfolio_df, CONFIG, report_level="Portfolio")
        
        strategy_name = CONFIG['strategy_name']
        log_folder = os.path.join(CONFIG['log_folder'], strategy_name)
        os.makedirs(log_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_path = os.path.join(log_folder, f"{timestamp}_portfolio_trade_log.csv")
        summary_file_path = os.path.join(log_folder, f"{timestamp}_summary.txt")
        
        portfolio_df.to_csv(log_file_path, index=False)
        print(f"\nCombined portfolio trade log saved to: {log_file_path}")
        save_config_summary(CONFIG, summary_file_path)
    else:
        print("\nNo trades were executed across any symbols.")
