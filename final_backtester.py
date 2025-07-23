import pandas as pd
import os
import math
from datetime import datetime
import time
from collections import defaultdict

# Configuration
INITIAL_CAPITAL = 1000000
RISK_PER_TRADE_PERCENT = 4.0
DATA_FOLDER = 'daily_with_indicators'
LOG_FOLDER = 'backtest_logs'
START_DATE = '2020-01-01'
END_DATE = '2025-07-16'

def parse_datetime(dt_str):
    formats = [
        '%d-%m-%Y %H:%M',
        '%Y-%m-%d %H:%M:%S'
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Time data '{dt_str}' doesn't match any expected format")

def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 2
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def calculate_metrics(trades):
    if not trades:
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0
        }
    
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    
    win_rate = len(winning_trades)/len(trades) if trades else 0
    profit_factor = (sum(t['pnl'] for t in winning_trades)/abs(sum(t['pnl'] for t in losing_trades))) if losing_trades else math.inf
    
    equity_curve = [INITIAL_CAPITAL]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade['pnl'])
    
    peak = equity_curve[0]
    max_drawdown = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value)/peak
        if dd > max_drawdown:
            max_drawdown = dd
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': sum(t['pnl'] for t in winning_trades)/len(winning_trades) if winning_trades else 0,
        'avg_loss': sum(t['pnl'] for t in losing_trades)/len(losing_trades) if losing_trades else 0,
        'max_drawdown': max_drawdown
    }

def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(LOG_FOLDER, exist_ok=True)
    
    print("Starting backtest...")
    print(f"Time: {timestamp}")
    print(f"Parameters: Risk={RISK_PER_TRADE_PERCENT}%, Date Range={START_DATE} to {END_DATE}")
    
    # Load symbols
    symbols = pd.read_csv('nifty200.csv')['Symbol'].tolist()
    print(f"\nLoaded {len(symbols)} symbols")
    
    # Initialize tracking
    missed_trades = {
        'due_to_capital': [],
        'due_to_risk': [],
        'valid_setups': 0
    }
    setup_stats = defaultdict(int)
    
    # Load and preprocess data
    print("\nProcessing stock data...")
    stock_data = {}
    for symbol in symbols:
        file_path = os.path.join(DATA_FOLDER, f"{symbol}_daily_with_indicators.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df.columns = [col.lower() for col in df.columns]
                df['datetime'] = df['datetime'].apply(parse_datetime)
                df = df[(df['datetime'] >= pd.Timestamp(START_DATE)) & 
                        (df['datetime'] <= pd.Timestamp(END_DATE))]
                
                if not df.empty:
                    df.set_index('datetime', inplace=True)
                    df.sort_index(inplace=True)
                    df['red_candle'] = df['close'] < df['open']
                    df['green_candle'] = df['close'] > df['open']
                    df['ema_30'] = df['ema_30'].ffill()
                    stock_data[symbol] = df
                    print(f"Processed {symbol}", end='\r')
            except Exception as e:
                print(f"\nError processing {symbol}: {str(e)}")
    print(f"\nFinished processing {len(stock_data)} symbols")

    if not stock_data:
        print("No valid stock data available")
        return
    
    # Initialize portfolio
    portfolio = {
        'cash': INITIAL_CAPITAL,
        'equity': INITIAL_CAPITAL,
        'positions': {},
        'trades': [],
        'daily_values': [INITIAL_CAPITAL]
    }
    
    # Main backtest loop
    all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))
    total_days = len(all_dates)
    print(f"\nRunning backtest for {total_days} days...")
    
    for i, date in enumerate(all_dates):
        exit_proceeds = 0
        to_remove = []
        
        # Print progress every 10 days or on last day
        if (i + 1) % 10 == 0 or (i + 1) == total_days:
            progress = (i + 1)/total_days * 100
            print(f"Progress: {i+1}/{total_days} days ({progress:.1f}%) | Equity: {portfolio['equity']:,.2f}", end='\r')
        
        # Process exits
        for pos_id, position in portfolio['positions'].items():
            symbol = position['symbol']
            if date not in stock_data[symbol].index:
                continue
                
            data = stock_data[symbol].loc[date]
            risk = position['entry_price'] - position['stop_loss']
            
            # Partial profit exit (1:1)
            if not position['partial_exit'] and data['high'] >= position['target']:
                sell_shares = position['shares'] // 2
                exit_value = sell_shares * position['target']
                exit_proceeds += exit_value
                
                portfolio['trades'].append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'].date(),
                    'exit_date': date.date(),
                    'pnl': (position['target'] - position['entry_price']) * sell_shares,
                    'exit_type': 'Partial Profit (1:1)'
                })
                
                position['shares'] -= sell_shares
                position['partial_exit'] = True
            
            # Stop loss exit
            if position['shares'] > 0 and data['low'] <= position['stop_loss']:
                exit_value = position['shares'] * position['stop_loss']
                exit_proceeds += exit_value
                
                portfolio['trades'].append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'].date(),
                    'exit_date': date.date(),
                    'pnl': (position['stop_loss'] - position['entry_price']) * position['shares'],
                    'exit_type': 'Stop-Loss'
                })
                
                to_remove.append(pos_id)
            
            # Update trailing stop
            elif position['shares'] > 0 and data['close'] > position['entry_price'] and data['green_candle']:
                position['stop_loss'] = max(position['stop_loss'], data['low'])
        
        # Remove exited positions
        for pos_id in to_remove:
            portfolio['positions'].pop(pos_id, None)
        
        # Store start of day values
        start_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in stock_data[pos['symbol']].index:
                start_equity += pos['shares'] * stock_data[pos['symbol']].loc[date]['close']
        
        # Process entries with detailed capital tracking
        for symbol, df in stock_data.items():
            if date not in df.index:
                continue
                
            # Skip if already in a position
            if any(pos['symbol'] == symbol for pos in portfolio['positions'].values()):
                continue
            
            try:
                loc = df.index.get_loc(date)
                if loc < 3:
                    continue
                
                # Get candles
                current = df.iloc[loc]
                prev1 = df.iloc[loc-1]
                
                # Must have at least one red candle before the green
                if not (prev1['green_candle'] and df.iloc[loc-2]['red_candle']):
                    continue
                
                # Count all valid setups
                missed_trades['valid_setups'] += 1
                pattern_length = len(get_consecutive_red_candles(df, loc)) + 1
                setup_stats[f'{pattern_length}R_1G'] += 1
                
                # EMA condition
                if not (prev1['close'] > prev1['ema_30']):
                    continue
                
                # Calculate entry price
                red_candles = get_consecutive_red_candles(df, loc)
                entry_price = max([c['high'] for c in red_candles] + [prev1['high']])
                
                # Check trigger
                if not (current['open'] < entry_price and current['high'] >= entry_price):
                    continue
                
                # Risk calculation (5-day lookback)
                stop_loss = df.iloc[loc-5:loc]['low'].min()
                risk = entry_price - stop_loss
                
                if risk <= 0:
                    missed_trades['due_to_risk'].append({
                        'date': date,
                        'symbol': symbol,
                        'reason': 'Invalid risk calculation'
                    })
                    continue
                
                # Position sizing
                risk_capital = start_equity * (RISK_PER_TRADE_PERCENT / 100)
                shares = math.floor(risk_capital / risk)
                
                if shares == 0:
                    missed_trades['due_to_capital'].append({
                        'date': date,
                        'symbol': symbol,
                        'required': entry_price * 1,
                        'available': portfolio['cash'],
                        'reason': 'Zero shares calculated'
                    })
                    continue
                
                if (shares * entry_price) > portfolio['cash']:
                    missed_trades['due_to_capital'].append({
                        'date': date,
                        'symbol': symbol,
                        'required': shares * entry_price,
                        'available': portfolio['cash'],
                        'reason': 'Insufficient capital'
                    })
                    # Attempt partial position
                    shares = math.floor(portfolio['cash'] / entry_price)
                    if shares == 0:
                        continue
                
                # Execute trade
                portfolio['cash'] -= shares * entry_price
                portfolio['positions'][f"{symbol}_{date}"] = {
                    'symbol': symbol,
                    'entry_date': date,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': entry_price + risk,
                    'shares': shares,
                    'partial_exit': False
                }
                
            except Exception as e:
                print(f"\nError processing {symbol} on {date}: {str(e)}")
        
        # Update portfolio
        portfolio['cash'] += exit_proceeds
        
        # Calculate end of day value
        end_value = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in stock_data[pos['symbol']].index:
                end_value += pos['shares'] * stock_data[pos['symbol']].loc[date]['close']
        
        portfolio['equity'] = end_value
        portfolio['daily_values'].append(end_value)
    
    # Calculate performance metrics
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    cagr = (portfolio['equity']/INITIAL_CAPITAL)**(1/years) - 1
    metrics = calculate_metrics(portfolio['trades'])
    total_trades = len(portfolio['trades'])
    taken_rate = total_trades/missed_trades['valid_setups'] if missed_trades['valid_setups'] else 0
    
    # Print final results to console
    print("\n\nBacktest Complete!")
    print("=================")
    print(f"Duration: {time.time()-start_time:.2f} seconds")
    print(f"Final Equity: {portfolio['equity']:,.2f}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Total Trades: {total_trades}")
    print(f"Valid Setups: {missed_trades['valid_setups']}")
    print(f"Missed Due to Capital: {len(missed_trades['due_to_capital'])}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Save detailed reports
    trades_df = pd.DataFrame(portfolio['trades'])
    trades_df.to_csv(
        os.path.join(LOG_FOLDER, f"{timestamp}_trades_detail.csv"), 
        index=False
    )
    pd.DataFrame(missed_trades['due_to_capital']).to_csv(
        os.path.join(LOG_FOLDER, f"{timestamp}_missed_trades_capital.csv"),
        index=False
    )
    
    # Save summary report with strategy config
    with open(os.path.join(LOG_FOLDER, f"{timestamp}_summary_report.txt"), 'w') as f:
        f.write("STRATEGY CONFIGURATION\n")
        f.write("======================\n")
        f.write(f"Initial Capital: {INITIAL_CAPITAL:,.2f}\n")
        f.write(f"Risk Per Trade: {RISK_PER_TRADE_PERCENT}%\n")
        f.write(f"EMA Period: 30\n")
        f.write(f"Stop Loss Lookback: 5 days\n")
        f.write(f"Partial Profit: 50% at 1:1 R:R\n")
        f.write(f"Trailing Stop: On green candles\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("===================\n")
        f.write(f"Final Equity: {portfolio['equity']:,.2f}\n")
        f.write(f"Net P&L: {portfolio['equity'] - INITIAL_CAPITAL:,.2f}\n")
        f.write(f"CAGR: {cagr:.2%}\n")
        f.write(f"Total Days: {len(all_dates)}\n")
        f.write(f"Valid Setups: {missed_trades['valid_setups']}\n")
        f.write(f"Trades Taken: {total_trades}\n")
        f.write(f"Execution Rate: {taken_rate:.1%}\n")
        f.write(f"Win Rate: {metrics['win_rate']:.1%}\n")
        f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
        f.write(f"Avg Win: {metrics['avg_win']:,.2f}\n")
        f.write(f"Avg Loss: {metrics['avg_loss']:,.2f}\n")
        f.write(f"Max Drawdown: {metrics['max_drawdown']:.1%}\n\n")
        
        f.write("MISSED TRADE ANALYSIS\n")
        f.write("=====================\n")
        f.write(f"Due to Capital: {len(missed_trades['due_to_capital'])}\n")
        if missed_trades['due_to_capital']:
            avg_shortfall = sum(t['required']-t['available'] for t in missed_trades['due_to_capital'])/len(missed_trades['due_to_capital'])
            f.write(f"Avg Capital Shortfall: {avg_shortfall:,.2f}\n")
        f.write(f"Due to Risk: {len(missed_trades['due_to_risk'])}\n\n")
        
        f.write("PATTERN STATISTICS\n")
        f.write("==================\n")
        for pattern, count in sorted(setup_stats.items()):
            f.write(f"{pattern}: {count} ({count/missed_trades['valid_setups']:.1%})\n")

if __name__ == "__main__":
    main()