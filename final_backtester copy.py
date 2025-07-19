import pandas as pd
import os
import math
from datetime import datetime
import time
import sys

# --- CONFIGURATION ---
# Centralized configuration for easy modification and reporting
config = {
    'initial_capital': 1000000,
    'risk_per_trade_percent': 3.0,
    'data_folder': 'daily_with_indicators',
    'log_folder': 'backtest_logs',
    'start_date': '2024-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',
    'ema_period': 30,
    'stop_loss_lookback': 5
}


def parse_datetime(dt_str):
    """Try multiple datetime formats to parse the string, focusing on timezone-naive formats."""
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%d-%m-%Y %H:%M',
        '%Y-%m-%d',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(dt_str).to_pydatetime().replace(tzinfo=None)
    except ValueError:
         raise ValueError(f"Time data '{dt_str}' doesn't match any expected format")


def get_consecutive_red_candles(df, current_loc):
    """Find all consecutive red candles before the green candle at T-1"""
    red_candles = []
    # The setup is based on T-1 being green, so the search for red
    # candles must start from T-2 (current_loc - 2).
    i = current_loc - 2 
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def run_backtest(cfg):
    """Main backtesting engine. Uses original logic with enhanced reporting."""
    start_time = time.time()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(cfg['log_folder'], exist_ok=True)
    
    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
        print(f"Loaded {len(symbols)} symbols from {cfg['nifty_list_csv']}")
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}")
        return

    stock_data = {}
    print("Loading and preprocessing data...")
    for symbol in symbols:
        file_path = os.path.join(cfg['data_folder'], f"{symbol}_daily_with_indicators.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                date_col_name = df.columns[0]
                df[date_col_name] = df[date_col_name].apply(str)
                df['datetime'] = df[date_col_name].apply(parse_datetime)
                
                df.columns = [col.lower() for col in df.columns]
                
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
                
                df = df.loc[cfg['start_date']:cfg['end_date']]

                if not df.empty:
                    df['red_candle'] = df['close'] < df['open']
                    df['green_candle'] = df['close'] > df['open']
                    
                    ema_col = f"ema_{cfg['ema_period']}"
                    if ema_col in df.columns:
                        df[ema_col] = df[ema_col].ffill()
                        stock_data[symbol] = df
                    else:
                        pass

            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
    
    if not stock_data:
        print("No valid stock data available for the given date range and parameters.")
        return
    print(f"Successfully processed data for {len(stock_data)} symbols.")

    portfolio = {
        'cash': cfg['initial_capital'],
        'equity': cfg['initial_capital'],
        'positions': {},
        'trades': [],
        'daily_values': []
    }
    missed_trades_capital = 0
    
    all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))
    
    print("Starting backtest simulation...")
    for i, date in enumerate(all_dates):
        progress_str = f"Processing {i+1}/{len(all_dates)}: {date.date()} | Equity: {portfolio['equity']:,.0f} | Open Positions: {len(portfolio['positions'])} | Trades: {len(portfolio['trades'])}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}")
        sys.stdout.flush()
        
        exit_proceeds = 0
        to_remove = []
        todays_exits = [] 
        
        # --- EXIT LOGIC ---
        for pos_id, position in list(portfolio['positions'].items()):
            symbol = position['symbol']
            if date not in stock_data[symbol].index:
                continue
                
            data = stock_data[symbol].loc[date]
            
            if not position['partial_exit'] and data['high'] >= position['target']:
                sell_shares = position['shares'] // 2
                exit_price = position['target']
                exit_value = sell_shares * exit_price
                exit_proceeds += exit_value
                
                todays_exits.append({
                    'symbol': symbol, 'entry_date': position['entry_date'].date(),
                    'exit_date': date.date(), 'entry_price': position['entry_price'],
                    'pnl': (exit_price - position['entry_price']) * sell_shares,
                    'exit_type': 'Partial Profit (1:1)',
                    'portfolio_equity_on_entry': position['portfolio_equity_on_entry'],
                    'risk_per_share': position['risk_per_share'],
                    'initial_shares': position['initial_shares'],
                    'initial_stop_loss': position['initial_stop_loss']
                })
                
                position['shares'] -= sell_shares
                position['partial_exit'] = True
                position['stop_loss'] = position['entry_price'] 

            if position['shares'] > 0 and data['low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_value = position['shares'] * exit_price
                exit_proceeds += exit_value
                
                todays_exits.append({
                    'symbol': symbol, 'entry_date': position['entry_date'].date(),
                    'exit_date': date.date(), 'entry_price': position['entry_price'],
                    'pnl': (exit_price - position['entry_price']) * position['shares'],
                    'exit_type': 'Stop-Loss',
                    'portfolio_equity_on_entry': position['portfolio_equity_on_entry'],
                    'risk_per_share': position['risk_per_share'],
                    'initial_shares': position['initial_shares'],
                    'initial_stop_loss': position['initial_stop_loss']
                })
                
                to_remove.append(pos_id)
                continue

            if position['shares'] > 0 and position['partial_exit'] and data['green_candle']:
                position['stop_loss'] = max(position['stop_loss'], data['low'])

        for pos_id in to_remove:
            portfolio['positions'].pop(pos_id, None)
        
        portfolio['cash'] += exit_proceeds

        # --- BUG FIX: Recalculate equity after exits and before new entries ---
        # This ensures risk calculations are based on the most current portfolio value.
        equity_after_exits = portfolio['cash']
        for pos in portfolio['positions'].values():
             if date in stock_data[pos['symbol']].index:
                equity_after_exits += pos['shares'] * stock_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = equity_after_exits
        # --- END OF BUG FIX ---

        # --- ENTRY LOGIC ---
        for symbol, df in stock_data.items():
            if any(pos['symbol'] == symbol for pos in portfolio['positions'].values()):
                continue
            
            if date not in df.index:
                continue
            
            try:
                loc = df.index.get_loc(date)
                if loc < 2: continue

                prev1 = df.iloc[loc-1]
                
                if not prev1['green_candle']:
                    continue
                
                if prev1['close'] < (prev1['high'] + prev1['low']) / 2:
                    continue

                ema_col = f"ema_{cfg['ema_period']}"
                if not (prev1['close'] > prev1[ema_col]):
                    continue
                
                red_candles = get_consecutive_red_candles(df, loc)
                if not red_candles:
                    continue
                
                highs_to_consider = [candle['high'] for candle in red_candles] + [prev1['high']]
                entry_trigger_price = max(highs_to_consider)
                
                today_candle = df.iloc[loc]

                if today_candle['high'] >= entry_trigger_price and today_candle['open'] < entry_trigger_price:
                    entry_price = entry_trigger_price
                    
                    sl_start_loc = max(0, loc - cfg['stop_loss_lookback'])
                    stop_loss = df.iloc[sl_start_loc:loc]['low'].min()
                    
                    risk_per_share = entry_price - stop_loss
                    if risk_per_share <= 0:
                        continue
                        
                    # Use the newly updated equity for risk calculation
                    equity_at_entry = portfolio['equity']
                    risk_capital = equity_at_entry * (cfg['risk_per_trade_percent'] / 100)
                    shares = math.floor(risk_capital / risk_per_share)
                    
                    if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                        portfolio['cash'] -= shares * entry_price
                        portfolio['positions'][f"{symbol}_{date}"] = {
                            'symbol': symbol, 'entry_date': date,
                            'entry_price': entry_price, 'stop_loss': stop_loss,
                            'shares': shares, 'target': entry_price + risk_per_share,
                            'partial_exit': False,
                            'portfolio_equity_on_entry': equity_at_entry,
                            'risk_per_share': risk_per_share,
                            'initial_shares': shares,
                            'initial_stop_loss': stop_loss
                        }
                    elif shares > 0:
                        missed_trades_capital += 1
            except Exception as e:
                pass
        
        # --- FINAL END OF DAY EQUITY CALCULATION (for logging) ---
        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in stock_data[pos['symbol']].index:
                eod_equity += pos['shares'] * stock_data[pos['symbol']].loc[date]['close']
        
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

        for exit_log in todays_exits:
            exit_log['portfolio_equity_on_exit'] = portfolio['equity']
            portfolio['trades'].append(exit_log)

    
    print("\n") 
    print("--- BACKTEST COMPLETE ---")
    
    final_equity = portfolio['equity']
    net_pnl = final_equity - cfg['initial_capital']
    
    equity_df = pd.DataFrame(portfolio['daily_values']).set_index('date')
    if not equity_df.empty:
        years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = ((final_equity / cfg['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100
    else:
        cagr, max_drawdown = 0, 0

    trades_df = pd.DataFrame(portfolio['trades'])
    total_trades, win_rate, profit_factor, avg_win, avg_loss = 0, 0, 0, 0, 0
    if not trades_df.empty:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        total_trades = len(trades_df)
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0

    summary_content = f"""BACKTEST SUMMARY REPORT
================================
INPUT PARAMETERS:
-----------------
Start Date: {cfg['start_date']}
End Date: {cfg['end_date']}
Initial Capital: {cfg['initial_capital']:,.2f}
Risk Per Trade: {cfg['risk_per_trade_percent']:.1f}%
EMA Period: {cfg['ema_period']}
Stop Loss Lookback: {cfg['stop_loss_lookback']} days

PERFORMANCE METRICS:
--------------------
Final Equity: {final_equity:,.2f}
Net P&L: {net_pnl:,.2f}
CAGR: {cagr:.2f}%
Max Drawdown: {max_drawdown:.1f}%

TRADE STATISTICS:
-----------------
Total Trade Events (incl. partials): {total_trades}
Win Rate (of events): {win_rate:.1f}%
Profit Factor: {profit_factor:.2f}
Average Winning Event: {avg_win:,.2f}
Average Losing Event: {avg_loss:,.2f}
Missed Trades (Capital): {missed_trades_capital}
"""
    
    summary_filename = os.path.join(cfg['log_folder'], f"{timestamp}_summary_report.txt")
    trades_filename = os.path.join(cfg['log_folder'], f"{timestamp}_trades_detail.csv")
    
    with open(summary_filename, 'w') as f:
        f.write(summary_content)
    
    if not trades_df.empty:
        log_columns = [
            'symbol', 'entry_date', 'exit_date', 'entry_price', 'pnl', 'exit_type',
            'portfolio_equity_on_entry', 'portfolio_equity_on_exit', 
            'risk_per_share', 'initial_shares', 'initial_stop_loss'
        ]
        trades_df = trades_df[log_columns]
        trades_df.to_csv(trades_filename, index=False)
    else:
        with open(trades_filename, 'w') as f:
            f.write("symbol,entry_date,exit_date,entry_price,pnl,exit_type,portfolio_equity_on_entry,portfolio_equity_on_exit,risk_per_share,initial_shares,initial_stop_loss\n")

    
    print(summary_content)
    print(f"Backtest completed in {time.time()-start_time:.2f} seconds")
    print(f"Reports saved to '{cfg['log_folder']}'")


if __name__ == "__main__":
    run_backtest(config)
