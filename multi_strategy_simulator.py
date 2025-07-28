# multi_strategy_simulator.py

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import configparser
import argparse
from pathlib import Path
import sys

class ConfigManager:
    """
    Manages loading and accessing configuration settings from an INI file.
    It merges a 'common' section with a strategy-specific section.
    """
    def __init__(self, config_path, strategy_section):
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        self.config.read(config_path)

        if 'common' not in self.config:
            raise ValueError("Config file must contain a [common] section.")
        if strategy_section not in self.config:
            raise ValueError(f"Strategy section '[{strategy_section}]' not found in config file.")

        self.settings = dict(self.config['common'])
        self.settings.update(dict(self.config[strategy_section]))
        # Store end_date as a datetime object for later comparisons
        self.settings['end_date_dt'] = pd.to_datetime(self.settings['end_date'])

    def get(self, key, fallback=None):
        return self.settings.get(key, fallback)

    def get_bool(self, key, fallback=False):
        return self.settings.get(key, str(fallback)).lower() in ('true', '1', 't', 'y', 'yes')

    def get_int(self, key, fallback=0):
        return int(self.settings.get(key, fallback))

    def get_float(self, key, fallback=0.0):
        return float(self.settings.get(key, fallback))

class Logger:
    """
    Handles detailed logging for backtest runs, creating strategy-specific directories
    and timestamped files within them. Console output is handled separately for progress display.
    """
    def __init__(self, config):
        self.log_path = Path(config.get('log_path', 'backtest_logs'))
        self.strategy_name = config.get('strategy_name', 'default_strategy').replace(' ', '_').lower()
        
        self.strategy_dir = self.log_path / self.strategy_name
        self.strategy_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        self.log_file = self.strategy_dir / f'{timestamp}_run.log'
        self.trades_summary_file = self.strategy_dir / f'{timestamp}_trades_summary.csv'

        # Configure root logger to write detailed logs to a file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(self.log_file)],
            force=True # Override any existing handlers
        )
        
        # Create a separate logger for console output for the final report
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.console_logger = logging.getLogger('console_logger')
        self.console_logger.addHandler(console_handler)
        self.console_logger.propagate = False

        logging.info(f"Logging initialized for strategy: {config.get('strategy_name')}")
        logging.info(f"Log files will be saved in: {self.strategy_dir}")

    def log_trade(self, trade_data):
        """Logs a completed trade to the summary CSV."""
        df = pd.DataFrame([trade_data])
        header = not os.path.exists(self.trades_summary_file)
        df.to_csv(self.trades_summary_file, index=False, header=header, mode='a')

class DataLoader:
    """
    Loads and prepares all necessary data for the backtest.
    """
    def __init__(self, config):
        self.data_path = Path(config.get('data_path'))
        self.index_symbol = config.get('index_symbol')
        self.stock_data = {}
        self.index_data = None

    def load_data(self):
        """Loads all stock and index data from the processed directory."""
        print("Loading data...")
        
        daily_data_path = self.data_path / 'daily'
        
        index_file = daily_data_path / f"{self.index_symbol}_daily_with_indicators.csv"
        if not index_file.exists():
            raise FileNotFoundError(f"Index data file not found: {index_file}")
        self.index_data = pd.read_csv(index_file, parse_dates=['datetime'])
        self.index_data.set_index('datetime', inplace=True)
        
        stock_files = daily_data_path.glob('*_daily_with_indicators.csv')
        for file in stock_files:
            symbol = file.name.replace('_daily_with_indicators.csv', '')
            if symbol != self.index_symbol:
                df = pd.read_csv(file, parse_dates=['datetime'])
                df.set_index('datetime', inplace=True)
                self.stock_data[symbol] = df
        
        print(f"Loaded data for {len(self.stock_data)} stocks and index {self.index_symbol}.")

class BacktestRunner:
    """
    Main class to orchestrate the backtesting process based on the selected strategy.
    """
    def __init__(self, config, logger, data_loader):
        self.config = config
        self.logger = logger
        self.data_loader = data_loader
        
        self.start_date = pd.to_datetime(config.get('start_date'))
        self.end_date = pd.to_datetime(config.get('end_date'))
        
        self.cash = config.get_float('initial_capital', 100000.0)
        self.risk_per_trade = config.get_float('risk_per_trade', 0.02) # Corrected from 2.0
        
        self.open_positions = {}
        self.trade_history = []
        self.daily_portfolio_value = []
        self.pending_signals = []

    def run(self):
        """Executes the backtest loop for the specified date range."""
        self.data_loader.load_data()
        
        strategy_class = self._get_strategy_class()
        strategy = strategy_class(self.config)

        dates = self.data_loader.index_data.loc[self.start_date:self.end_date].index

        for i, current_date in enumerate(dates):
            equity_for_risk_calc = self.daily_portfolio_value[-1]['value'] if self.daily_portfolio_value else self.cash

            self._manage_open_positions(current_date)

            signals_to_process = self.pending_signals
            self.pending_signals = [] 

            new_signals = strategy.generate_signals(
                current_date, 
                self.data_loader.stock_data, 
                self.data_loader.index_data
            )
            signals_to_process.extend(new_signals)

            self._execute_signals(signals_to_process, current_date, equity_for_risk_calc)
            
            self._log_daily_pnl(current_date)
            
            todays_equity = self.daily_portfolio_value[-1]['value']
            progress = (i + 1) / len(dates)
            progress_bar = f"[{'=' * int(progress * 20):<20}]"
            progress_str = f"Processing {current_date.date()} {progress_bar} {progress:.0%} | Equity: {todays_equity:,.0f} | Positions: {len(self.open_positions)}"
            sys.stdout.write(f"\r{progress_str.ljust(100)}")
            sys.stdout.flush()


        self._close_all_positions(dates[-1])
        self._generate_summary_report()

    def _get_strategy_class(self):
        strategy_map = {
            "eod_confirmation": EODConfirmationStrategy,
            "intraday_breakout": IntradayBreakoutStrategy,
            "intraday_confirmation": IntradayConfirmationStrategy,
        }
        strategy_key = self.config.get('strategy_key')
        if strategy_key not in strategy_map:
            raise ValueError(f"Unknown strategy key: {strategy_key}")
        return strategy_map[strategy_key]

    def _manage_open_positions(self, current_date):
        positions_to_close = []
        for symbol, trade in self.open_positions.items():
            stock_df = self.data_loader.stock_data.get(symbol)
            if stock_df is None or current_date not in stock_df.index:
                continue

            today_data = stock_df.loc[current_date]
            
            if today_data['low'] <= trade['stop_loss']:
                exit_price = min(today_data['open'], trade['stop_loss'])
                trade['exit_price'] = exit_price
                trade['exit_date'] = current_date
                trade['status'] = 'Closed (SL)'
                positions_to_close.append(symbol)
                continue

            if not trade['partial_profit_taken'] and today_data['high'] >= trade['profit_target']:
                trade['partial_profit_taken'] = True
                if self.config.get_bool('use_aggressive_breakeven'):
                    trade['stop_loss'] = trade['entry_price']
                logging.info(f"Partial profit taken for {symbol} on {current_date.date()}. Stop moved to breakeven.")

            if trade['partial_profit_taken']:
                if today_data['close'] > today_data['open']:
                    trade['stop_loss'] = max(trade['stop_loss'], today_data['low'])

        for symbol in positions_to_close:
            self._close_trade(symbol)

    def _execute_signals(self, signals, current_date, equity_for_risk_calc):
        for signal in signals:
            entry_date = signal['entry_date']
            
            if entry_date > current_date:
                self.pending_signals.append(signal)
                continue
            
            if entry_date == current_date:
                symbol = signal['symbol']
                if symbol in self.open_positions:
                    continue

                stock_df = self.data_loader.stock_data.get(symbol)
                if stock_df is None or entry_date not in stock_df.index:
                    continue
                
                entry_price = stock_df.loc[entry_date]['open']
                stop_loss = signal['stop_loss']
                
                if self.config.get_bool('use_gap_up_filter'):
                    setup_high = signal['setup_high']
                    if entry_price > setup_high * 1.02:
                        logging.info(f"Skipping {symbol} due to >2% gap up.")
                        continue

                risk_per_share = entry_price - stop_loss
                if risk_per_share <= 0:
                    continue
                
                risk_amount = equity_for_risk_calc * self.risk_per_trade
                quantity = int(risk_amount / risk_per_share)
                
                if quantity == 0:
                    continue

                trade_cost = entry_price * quantity
                if self.cash >= trade_cost:
                    self.cash -= trade_cost
                    self.open_positions[symbol] = {
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'stop_loss': stop_loss,
                        'profit_target': entry_price + risk_per_share,
                        'status': 'Open',
                        'partial_profit_taken': False,
                        'exit_date': None,
                        'exit_price': None,
                        'pnl': 0.0
                    }
                    logging.info(f"ENTERING TRADE: {symbol} on {entry_date.date()} at {entry_price:.2f}, Qty: {quantity}, SL: {stop_loss:.2f}")
                else:
                    logging.info(f"SKIPPING TRADE: Insufficient cash for {symbol}.")


    def _close_trade(self, symbol):
        trade = self.open_positions.pop(symbol)
        
        trade_proceeds = trade['exit_price'] * trade['quantity']
        self.cash += trade_proceeds
        
        trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['quantity']
        self.trade_history.append(trade)
        self.logger.log_trade(trade)
        logging.info(f"CLOSING TRADE: {symbol} on {trade['exit_date'].date()} at {trade['exit_price']:.2f}. P&L: {trade['pnl']:.2f}")

    def _log_daily_pnl(self, current_date):
        market_value_of_open_positions = 0
        for symbol, trade in self.open_positions.items():
            stock_df = self.data_loader.stock_data.get(symbol)
            if stock_df is not None and current_date in stock_df.index:
                current_price = stock_df.loc[current_date]['close']
                market_value_of_open_positions += current_price * trade['quantity']
        
        portfolio_value = self.cash + market_value_of_open_positions
        self.daily_portfolio_value.append({'datetime': current_date, 'value': portfolio_value})

    def _close_all_positions(self, last_date):
        symbols_to_close = list(self.open_positions.keys())
        for symbol in symbols_to_close:
            trade = self.open_positions[symbol]
            stock_df = self.data_loader.stock_data.get(symbol)
            if stock_df is not None and last_date in stock_df.index:
                trade['exit_price'] = stock_df.loc[last_date]['close']
                trade['exit_date'] = last_date
                trade['status'] = 'Closed (EOD)'
                self._close_trade(symbol)

    def _generate_summary_report(self):
        print("\n")
        report_header = "\n" + "="*50 + "\nBACKTEST SUMMARY REPORT\n" + "="*50
        self.logger.console_logger.info(report_header)
        logging.info(report_header)

        total_trades = len(self.trade_history)
        if total_trades == 0:
            msg = "No trades were executed."
            self.logger.console_logger.info(msg)
            logging.info(msg)
            return

        pnl_values = [t['pnl'] for t in self.trade_history]
        winning_trades = sum(1 for pnl in pnl_values if pnl > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_profit = sum(pnl for pnl in pnl_values if pnl > 0)
        total_loss = sum(pnl for pnl in pnl_values if pnl < 0)
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')

        portfolio_df = pd.DataFrame(self.daily_portfolio_value).set_index('datetime')
        initial_capital = self.config.get_float('initial_capital')
        
        if not portfolio_df.empty:
            end_capital = portfolio_df['value'].iloc[-1]
            total_return_pct = ((end_capital - initial_capital) / initial_capital) * 100
        else:
            end_capital = initial_capital
            total_return_pct = 0

        report_lines = [
            f"Total Trades: {total_trades}",
            f"Win Rate: {win_rate:.2f}%",
            f"Profit Factor: {profit_factor:.2f}",
            f"Total Return: {total_return_pct:.2f}%",
            f"Final Portfolio Value: ${end_capital:,.2f}",
            "="*50
        ]
        
        for line in report_lines:
            self.logger.console_logger.info(line)
            logging.info(line)

# --- Strategy Implementations ---

class BaseStrategy:
    def __init__(self, config):
        self.config = config
        self.ema_30_col = 'ema_30'
        self.ema_50_col = 'ema_50'
        self.avg_vol_20_col = 'volume_20_sma'
        self.rs_30_col = 'return_30'

    def generate_signals(self, current_date, stock_data, index_data):
        raise NotImplementedError

    def _find_setups(self, current_date, stock_data):
        setups = []
        
        for symbol, df in stock_data.items():
            try:
                loc = df.index.get_loc(current_date)
                if loc < 10: 
                    continue
                
                yesterday_loc = loc - 1
                yesterday = df.index[yesterday_loc]
                t_minus_1 = df.iloc[yesterday_loc]
                
                if t_minus_1['close'] <= t_minus_1['open']:
                    continue
                
                if self.ema_30_col not in t_minus_1 or pd.isna(t_minus_1[self.ema_30_col]) or t_minus_1['close'] < t_minus_1[self.ema_30_col]:
                    continue
                if t_minus_1['close'] >= (t_minus_1['high'] + t_minus_1['low']) / 2:
                    continue
                
                # --- FIX: Correctly identify preceding red candles ---
                red_candle_found = False
                for i in range(1, 6): # Look back up to 5 days
                    prev_loc = yesterday_loc - i
                    if prev_loc < 0:
                        break
                    if df.iloc[prev_loc]['close'] < df.iloc[prev_loc]['open']:
                        red_candle_found = True
                        break
                
                if not red_candle_found:
                    continue
                
                setups.append({'symbol': symbol, 'setup_date': yesterday, 'setup_high': t_minus_1['high']})

            except KeyError:
                continue
        return setups

class EODConfirmationStrategy(BaseStrategy):
    def generate_signals(self, current_date, stock_data, index_data):
        signals = []
        setups = self._find_setups(current_date, stock_data)
        
        if not setups:
            return signals

        if current_date not in index_data.index or self.ema_50_col not in index_data.columns or index_data.loc[current_date]['close'] < index_data.loc[current_date][self.ema_50_col]:
            return signals

        for setup in setups:
            symbol = setup['symbol']
            df = stock_data.get(symbol)
            if df is None or current_date not in df.index:
                continue
            
            today_data = df.loc[current_date]
            
            if today_data['high'] <= setup['setup_high']:
                continue
            
            if self.avg_vol_20_col not in today_data or pd.isna(today_data[self.avg_vol_20_col]) or today_data['volume'] < today_data[self.avg_vol_20_col] * 1.3:
                continue
            
            if self.rs_30_col not in today_data or self.rs_30_col not in index_data.columns or pd.isna(today_data[self.rs_30_col]) or pd.isna(index_data.loc[current_date][self.rs_30_col]) or today_data[self.rs_30_col] <= index_data.loc[current_date][self.rs_30_col]:
                continue
            
            entry_date = current_date + timedelta(days=1)
            while entry_date not in index_data.index:
                if entry_date > self.config.get('end_date_dt'):
                    entry_date = None
                    break
                entry_date += timedelta(days=1)
            
            if entry_date is None:
                continue

            sl_period = df.loc[:current_date].tail(5)
            stop_loss = sl_period['low'].min()
            
            if pd.isna(stop_loss):
                continue

            signals.append({
                'symbol': symbol,
                'entry_date': entry_date,
                'stop_loss': stop_loss,
                'setup_high': setup['setup_high']
            })
            logging.info(f"Signal for {symbol} on {current_date.date()} for entry on {entry_date.date()}.")
            
        return signals

class IntradayBreakoutStrategy(BaseStrategy):
    def generate_signals(self, current_date, stock_data, index_data):
        signals = []
        setups = self._find_setups(current_date, stock_data)
        
        if not setups:
            return signals

        yesterday = current_date - timedelta(days=1) # Note: This strategy might need the same robust date logic
        
        if yesterday not in index_data.index or self.ema_50_col not in index_data.columns or index_data.loc[yesterday]['close'] < index_data.loc[yesterday][self.ema_50_col]:
            return signals

        for setup in setups:
            symbol = setup['symbol']
            df = stock_data.get(symbol)
            if df is None or current_date not in df.index or setup['setup_date'] not in df.index:
                continue
            
            setup_date = setup['setup_date']
            if self.rs_30_col not in df.columns or self.rs_30_col not in index_data.columns or pd.isna(df.loc[setup_date][self.rs_30_col]) or pd.isna(index_data.loc[setup_date][self.rs_30_col]) or df.loc[setup_date][self.rs_30_col] <= index_data.loc[setup_date][self.rs_30_col]:
                continue

            if df.loc[current_date]['high'] > setup['setup_high']:
                sl_period = df.loc[:setup_date].tail(5)
                stop_loss = sl_period['low'].min()

                if pd.isna(stop_loss):
                    continue
                
                signals.append({
                    'symbol': symbol,
                    'entry_date': current_date,
                    'stop_loss': stop_loss,
                    'setup_high': setup['setup_high']
                })
                logging.info(f"Signal for {symbol} on {current_date.date()}.")

        return signals

class IntradayConfirmationStrategy(BaseStrategy):
    def generate_signals(self, current_date, stock_data, index_data):
        signals = []
        setups = self._find_setups(current_date, stock_data)
        
        if not setups:
            return signals

        yesterday = current_date - timedelta(days=1) # Note: This strategy might need the same robust date logic
        
        if yesterday not in index_data.index or self.ema_50_col not in index_data.columns or index_data.loc[yesterday]['close'] < index_data.loc[yesterday][self.ema_50_col]:
            return signals

        for setup in setups:
            symbol = setup['symbol']
            df = stock_data.get(symbol)
            if df is None or current_date not in df.index or setup['setup_date'] not in df.index:
                continue
            
            setup_date = setup['setup_date']
            if self.rs_30_col not in df.columns or self.rs_30_col not in index_data.columns or pd.isna(df.loc[setup_date][self.rs_30_col]) or pd.isna(index_data.loc[setup_date][self.rs_30_col]) or df.loc[setup_date][self.rs_30_col] <= index_data.loc[setup_date][self.rs_30_col]:
                continue

            if df.loc[current_date]['high'] > setup['setup_high']:
                sl_period = df.loc[:setup_date].tail(5)
                stop_loss = sl_period['low'].min()

                if pd.isna(stop_loss):
                    continue
                
                signals.append({
                    'symbol': symbol,
                    'entry_date': current_date,
                    'stop_loss': stop_loss,
                    'setup_high': setup['setup_high']
                })
                logging.info(f"Signal for {symbol} on {current_date.date()}.")
        
        return signals

def create_config_file():
    config_path = 'config.ini'
    if not os.path.exists(config_path):
        config = configparser.ConfigParser()
        config['common'] = {
            'data_path': './data/processed',
            'log_path': './backtest_logs',
            'index_symbol': 'NIFTY200_INDEX',
            'start_date': '2022-01-01',
            'end_date': '2023-12-31',
            'initial_capital': '100000',
            'risk_per_trade': '0.02', # Corrected from 2.0
            'use_gap_up_filter': 'True',
            'use_aggressive_breakeven': 'True'
        }
        config['eod_confirmation'] = {
            'strategy_key': 'eod_confirmation',
            'strategy_name': 'EOD Confirmation (T+1 Entry)'
        }
        config['intraday_breakout'] = {
            'strategy_key': 'intraday_breakout',
            'strategy_name': 'Intraday Breakout (T Day Entry)'
        }
        config['intraday_confirmation'] = {
            'strategy_key': 'intraday_confirmation',
            'strategy_name': 'Intraday Confirmation (T Day Entry w 3:15 Check)'
        }
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        print(f"Created default '{config_path}' file.")

if __name__ == '__main__':
    create_config_file()

    parser = argparse.ArgumentParser(description="Run a backtest for a specific trading strategy.")
    parser.add_argument('--strategy', type=str, required=True, 
                        choices=['eod_confirmation', 'intraday_breakout', 'intraday_confirmation'],
                        help='The name of the strategy section in config.ini to run.')
    args = parser.parse_args()

    try:
        config_manager = ConfigManager('config.ini', args.strategy)
        logger = Logger(config_manager)
        data_loader = DataLoader(config_manager)
        
        runner = BacktestRunner(config_manager, logger, data_loader)
        runner.run()
    except (FileNotFoundError, ValueError, AttributeError, KeyError) as e:
        print("\n")
        logging.error(f"A critical error occurred: {e}", exc_info=True)
