import pandas as pd
import os
import math
from datetime import datetime
import time

# Configuration
INITIAL_CAPITAL = 1000000
RISK_PER_TRADE_PERCENT = 5.0  # Risk 1% of capital per trade
DATA_FOLDER = 'daily_with_indicators'
LOG_FOLDER = 'backtest_logs'
START_DATE = '2020-01-01'
END_DATE = '2025-07-16'

def parse_datetime(dt_str):
    """Try multiple datetime formats to parse the string"""
    formats = [
        '%d-%m-%Y %H:%M',  # Original format (02-01-2023 05:30)
        '%Y-%m-%d %H:%M:%S'  # ISO format (2023-01-02 05:30:00)
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Time data '{dt_str}' doesn't match any expected format")

def get_consecutive_red_candles(df, current_loc):
    """Find all consecutive red candles before the green candle at T-1"""
    red_candles = []
    # Start from T-2 (since T-1 must be green)
    i = current_loc - 2
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def main():
    start_time = time.time()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(LOG_FOLDER, exist_ok=True)
    
    # Load symbols
    symbols = pd.read_csv('nifty200.csv')['Symbol'].tolist() #nifty50-Copy.csv
    print(f"Loaded {len(symbols)} symbols")
    
    # Load and preprocess data
    stock_data = {}
    for symbol in symbols:
        file_path = os.path.join(DATA_FOLDER, f"{symbol}_daily_with_indicators.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df.columns = [col.lower() for col in df.columns]
                
                # Parse datetime with flexible format handling
                df['datetime'] = df['datetime'].apply(parse_datetime)
                
                # Filter by date range
                df = df[(df['datetime'] >= pd.Timestamp(START_DATE)) & 
                        (df['datetime'] <= pd.Timestamp(END_DATE))]
                
                if not df.empty:
                    df.set_index('datetime', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Calculate required fields
                    df['red_candle'] = df['close'] < df['open']
                    df['green_candle'] = df['close'] > df['open']
                    df['ema_20'] = df['ema_20'].ffill()  # Fill forward missing EMAs
                    
                    stock_data[symbol] = df
                    print(f"Processed {symbol}")
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
    
    if not stock_data:
        print("No valid stock data available")
        return
    
    # Initialize portfolio
    portfolio = {
        'cash': INITIAL_CAPITAL,
        'equity': INITIAL_CAPITAL,
        'positions': {},
        'trades': [],
        'daily_values': []
    }
    
    # Get all trading days
    all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))
    
    # Main backtest loop
    for i, date in enumerate(all_dates):
        if (i + 1) % 10 == 0 or (i + 1) == len(all_dates):
            print(f"Processing {i+1}/{len(all_dates)}: {date.date()} | Equity: {portfolio['equity']:,.0f}")
        
        # Store start of day values
        start_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in stock_data[pos['symbol']].index:
                start_equity += pos['shares'] * stock_data[pos['symbol']].loc[date]['close']
        
        # Process exits
        exit_proceeds = 0
        to_remove = []
        
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
        
        # Process entries
        for symbol, df in stock_data.items():
            if date not in df.index:
                continue
                
            # Skip if already in a position
            if any(pos['symbol'] == symbol for pos in portfolio['positions'].values()):
                continue
            
            try:
                loc = df.index.get_loc(date)
                if loc < 3:  # Need at least 3 previous days for stop calculation
                    continue
                
                # Get current and previous candles
                current = df.iloc[loc]
                prev1 = df.iloc[loc-1]  # T-1 (must be green)
                
                # Must have at least one red candle before the green
                if not prev1['green_candle']:
                    continue
                
                # Get all consecutive red candles before the green
                red_candles = get_consecutive_red_candles(df, loc)
                if not red_candles:
                    continue
                
                # Check EMA condition on green candle
                if not (prev1['close'] > prev1['ema_20']):
                    continue
                
                # Calculate entry trigger price (max of all red candles' highs and green candle's high)
                highs_to_consider = [candle['high'] for candle in red_candles] + [prev1['high']]
                entry_trigger_price = max(highs_to_consider)
                
                # Check if current candle triggers entry
                if current['open'] < entry_trigger_price and current['high'] >= entry_trigger_price:
                    entry_price = entry_trigger_price
                    # Stop loss is min of last 3 lows (as per previous change)
                    stop_loss = df.iloc[loc-5:loc]['low'].min()
                    risk = entry_price - stop_loss
                    
                    if risk <= 0:
                        continue
                        
                    # Position sizing
                    risk_capital = start_equity * (RISK_PER_TRADE_PERCENT / 100)
                    shares = math.floor(risk_capital / risk)
                    
                    if shares > 0 and (shares * entry_price) <= portfolio['cash']:
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
                print(f"Error processing {symbol} on {date}: {str(e)}")
        
        # Update portfolio
        portfolio['cash'] += exit_proceeds
        
        # Calculate end of day value
        end_value = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in stock_data[pos['symbol']].index:
                end_value += pos['shares'] * stock_data[pos['symbol']].loc[date]['close']
        
        portfolio['equity'] = end_value
        portfolio['daily_values'].append(end_value)
    
    # Generate reports
    final_equity = portfolio['equity']
    net_pnl = final_equity - INITIAL_CAPITAL
    total_return = (net_pnl / INITIAL_CAPITAL) * 100
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    cagr = ((final_equity / INITIAL_CAPITAL) ** (1/years) - 1 if years > 0 else 0)
    
    # Save trades log
    trades_df = pd.DataFrame(portfolio['trades'])
    trades_df.to_csv(os.path.join(LOG_FOLDER, f"{timestamp}_trades_log.csv"), index=False)
    
    # Save summary
    with open(os.path.join(LOG_FOLDER, f"{timestamp}_summary_report.txt"), 'w') as f:
        f.write("Backtest Summary\n")
        f.write("===============\n\n")
        f.write(f"Initial Capital: {INITIAL_CAPITAL:,.2f}\n")
        f.write(f"Final Equity: {final_equity:,.2f}\n")
        f.write(f"Net P&L: {net_pnl:,.2f}\n")
        f.write(f"Total Return: {total_return:.2f}%\n")
        f.write(f"CAGR: {cagr:.2%}\n")
        f.write(f"Period: {years:.2f} years\n")
        f.write(f"Total Trades: {len(portfolio['trades'])}\n")
    
    print(f"\nBacktest completed in {time.time()-start_time:.2f} seconds")
    print(f"Final Equity: {final_equity:,.2f}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Reports saved to {LOG_FOLDER}")

if __name__ == "__main__":
    main()