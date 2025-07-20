import pandas as pd
import os
import math
from datetime import datetime, timedelta
import time
import sys

# --- CONFIGURATION ---
config = {
    'initial_capital': 1000000,
    'risk_per_trade_percent': 4.0,
    # Possible values: 'daily', '2day', 'weekly', 'monthly'
    'timeframe': 'weekly', 
    'data_folder_base': 'data/processed',
    'log_folder': 'backtest_logs',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',
    'ema_period': 30,
    'stop_loss_lookback': 5,
    # --- FILTERS ---
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200',
    'regime_ma_period': 50,
    'volume_filter': True,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    'atr_filter': False,
    'atr_period': 14,
    'atr_ma_period': 30,
    'atr_multiplier': 1.4,
    'rs_filter': True,
    'rs_index_symbol': 'NIFTY200',
    'rs_period': 30, 
    'rs_outperformance_pct': 0.0
}


def parse_datetime(dt_str):
    formats = ['%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M', '%Y-%m-%d']
    for fmt in formats:
        try: return datetime.strptime(dt_str, fmt)
        except ValueError: continue
    try: return pd.to_datetime(dt_str).to_pydatetime().replace(tzinfo=None)
    except ValueError: raise ValueError(f"Time data '{dt_str}' doesn't match any expected format")


def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 2 
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def run_backtest(cfg):
    start_time = time.time()
    data_folder = os.path.join(cfg['data_folder_base'], cfg['timeframe'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(cfg['log_folder'], exist_ok=True)
    
    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
        print(f"Loaded {len(symbols)} symbols from {cfg['nifty_list_csv']}")
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}")
        return

    # --- Load Index Data and Set Master Clock for Backtest Loop ---
    index_df_daily = None
    all_dates = []
    try:
        daily_index_folder = os.path.join(cfg['data_folder_base'], 'daily')
        index_filename = f"{cfg['regime_index_symbol']}_INDEX_daily_with_indicators.csv"
        index_path = os.path.join(daily_index_folder, index_filename)
        index_df_daily = pd.read_csv(index_path, index_col=0, parse_dates=True)
        index_df_daily.columns = [col.lower() for col in index_df_daily.columns]
        
        if cfg['rs_filter']:
            index_df_daily['return'] = index_df_daily['close'].pct_change(periods=cfg['rs_period']) * 100
        
        print(f"Successfully loaded Daily Index data from {index_path}")

        # Create the master date loop from the resampled index to ensure consistency
        if cfg['timeframe'] == 'daily':
            all_dates = index_df_daily.loc[cfg['start_date']:cfg['end_date']].index
        else:
            timeframe_rules = {'2day': '2D', 'weekly': 'W-MON', 'monthly': 'MS'}
            resampled_index = index_df_daily.resample(timeframe_rules[cfg['timeframe']]).last()
            all_dates = resampled_index.loc[cfg['start_date']:cfg['end_date']].index
        
        print(f"Master date loop created for '{cfg['timeframe']}' timeframe with {len(all_dates)} periods.")

    except FileNotFoundError:
        print(f"Error: Index file not found at {index_path}. Disabling filters.")
        cfg['market_regime_filter'] = False
        cfg['rs_filter'] = False

    stock_data = {}
    print(f"Loading and preprocessing data from '{data_folder}'...")
    for symbol in symbols:
        file_path = os.path.join(data_folder, f"{symbol}_daily_with_indicators.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df.columns = [col.lower() for col in df.columns]
                df = df.loc[cfg['start_date']:cfg['end_date']]
                if not df.empty:
                    df['red_candle'] = df['close'] < df['open']
                    df['green_candle'] = df['close'] > df['open']
                    if cfg['volume_filter']: df['volume_ma'] = df['volume'].rolling(window=cfg['volume_ma_period']).mean()
                    if cfg['rs_filter']: df['return'] = df['close'].pct_change(periods=cfg['rs_period']) * 100
                    stock_data[symbol] = df
            except Exception as e: print(f"Error processing {symbol}: {str(e)}")
    
    if not stock_data: print("No valid stock data available."); return
    print(f"Successfully processed data for {len(stock_data)} symbols.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    missed_trades_capital = 0
    
    print("Starting backtest simulation...")
    for i, date in enumerate(all_dates):
        progress_str = f"Processing {i+1}/{len(all_dates)}: {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])} | Trades: {len(portfolio['trades'])}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()
        
        exit_proceeds = 0; to_remove = []; todays_exits = [] 
        for pos_id, pos in list(portfolio['positions'].items()):
            symbol = pos['symbol']
            if date not in stock_data[symbol].index: continue
            data = stock_data[symbol].loc[date]
            if not pos['partial_exit'] and data['high'] >= pos['target']:
                shares = pos['shares'] // 2; price = pos['target']; exit_proceeds += shares * price
                todays_exits.append({'symbol': symbol, 'entry_date': pos['entry_date'].date(), 'exit_date': date.date(), 'entry_price': pos['entry_price'], 'pnl': (price - pos['entry_price']) * shares, 'exit_type': 'Partial Profit (1:1)', **pos})
                pos['shares'] -= shares; pos['partial_exit'] = True; pos['stop_loss'] = pos['entry_price'] 
            if pos['shares'] > 0 and data['low'] <= pos['stop_loss']:
                price = pos['stop_loss']; exit_proceeds += pos['shares'] * price
                todays_exits.append({'symbol': symbol, 'entry_date': pos['entry_date'].date(), 'exit_date': date.date(), 'entry_price': pos['entry_price'], 'pnl': (price - pos['entry_price']) * pos['shares'], 'exit_type': 'Stop-Loss', **pos})
                to_remove.append(pos_id); continue
            if pos['shares'] > 0 and data['close'] > pos['entry_price']:
                pos['stop_loss'] = max(pos['stop_loss'], pos['entry_price'])
                if data['green_candle']: pos['stop_loss'] = max(pos['stop_loss'], data['low'])
        for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
        portfolio['cash'] += exit_proceeds
        equity_after_exits = portfolio['cash']
        for pos in portfolio['positions'].values():
             if date in stock_data[pos['symbol']].index: equity_after_exits += pos['shares'] * stock_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = equity_after_exits

        market_uptrend = True
        if cfg['market_regime_filter'] and date in index_df_daily.index:
            if index_df_daily.loc[date]['close'] < index_df_daily.loc[date][f"ema_{cfg['regime_ma_period']}"]: market_uptrend = False
        
        if market_uptrend:
            for symbol, df in stock_data.items():
                if any(pos['symbol'] == symbol for pos in portfolio['positions'].values()): continue
                if date not in df.index: continue
                try:
                    loc = df.index.get_loc(date)
                    if loc < max(cfg['rs_period'], 2): continue
                    prev1 = df.iloc[loc-1]
                    if not prev1['green_candle'] or prev1['close'] < (prev1['high'] + prev1['low']) / 2: continue
                    if not (prev1['close'] > prev1[f"ema_{cfg['ema_period']}"]): continue
                    if not get_consecutive_red_candles(df, loc): continue
                    
                    rs_ok = True
                    if cfg['rs_filter'] and date in index_df_daily.index:
                        stock_return = df.loc[date, 'return']
                        index_return = index_df_daily.loc[date, 'return']
                        if pd.isna(stock_return) or pd.isna(index_return) or (stock_return < index_return + cfg['rs_outperformance_pct']):
                            rs_ok = False
                    if not rs_ok: continue

                    today_candle = df.iloc[loc]
                    volume_ok = not cfg['volume_filter'] or (pd.notna(today_candle['volume_ma']) and today_candle['volume'] >= (today_candle['volume_ma'] * cfg['volume_multiplier']))
                    atr_ok = not cfg['atr_filter'] or (pd.notna(today_candle[f"atr_{cfg['atr_period']}"]) and pd.notna(today_candle[f"atr_ma_{cfg['atr_ma_period']}"]) and today_candle[f"atr_{cfg['atr_period']}"] <= (today_candle[f"atr_ma_{cfg['atr_ma_period']}"] * cfg['atr_multiplier']))
                    
                    entry_trigger_price = max([c['high'] for c in get_consecutive_red_candles(df, loc)] + [prev1['high']])
                    if volume_ok and atr_ok and today_candle['high'] >= entry_trigger_price and today_candle['open'] < entry_trigger_price:
                        entry_price = entry_trigger_price
                        stop_loss = df.iloc[max(0, loc - cfg['stop_loss_lookback']):loc]['low'].min()
                        risk_per_share = entry_price - stop_loss
                        if risk_per_share <= 0: continue
                        equity_at_entry = portfolio['equity']
                        shares = math.floor((equity_at_entry * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)
                        if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                            portfolio['cash'] -= shares * entry_price
                            portfolio['positions'][f"{symbol}_{date}"] = {'symbol': symbol, 'entry_date': date, 'entry_price': entry_price, 'stop_loss': stop_loss, 'shares': shares, 'target': entry_price + risk_per_share, 'partial_exit': False, 'portfolio_equity_on_entry': equity_at_entry, 'risk_per_share': risk_per_share, 'initial_shares': shares, 'initial_stop_loss': stop_loss}
                        elif shares > 0: missed_trades_capital += 1
                except Exception: pass
        
        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in stock_data[pos['symbol']].index: eod_equity += pos['shares'] * stock_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})
        for exit_log in todays_exits:
            exit_log['portfolio_equity_on_exit'] = portfolio['equity']
            portfolio['trades'].append(exit_log)

    print("\n\n--- BACKTEST COMPLETE ---")
    final_equity = portfolio['equity']
    net_pnl = final_equity - cfg['initial_capital']
    equity_df = pd.DataFrame(portfolio['daily_values']).set_index('date')
    if not equity_df.empty:
        years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = ((final_equity / cfg['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
        peak = equity_df['equity'].cummax(); drawdown = (equity_df['equity'] - peak) / peak; max_drawdown = abs(drawdown.min()) * 100
    else: cagr, max_drawdown = 0, 0
    trades_df = pd.DataFrame(portfolio['trades'])
    total_trades, win_rate, profit_factor, avg_win, avg_loss = 0, 0, 0, 0, 0
    if not trades_df.empty:
        winning_trades = trades_df[trades_df['pnl'] > 0]; losing_trades = trades_df[trades_df['pnl'] <= 0]
        total_trades = len(trades_df)
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        gross_profit = winning_trades['pnl'].sum(); gross_loss = abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0

    summary_content = f"""BACKTEST SUMMARY REPORT
================================
INPUT PARAMETERS:
-----------------
Timeframe: {cfg['timeframe']}
Start Date: {cfg['start_date']}
End Date: {cfg['end_date']}
Initial Capital: {cfg['initial_capital']:,.2f}
Risk Per Trade: {cfg['risk_per_trade_percent']:.1f}%
EMA Period: {cfg['ema_period']}
Stop Loss Lookback: {cfg['stop_loss_lookback']} days
Market Regime Filter: {'Enabled' if cfg['market_regime_filter'] else 'Disabled'}
Volume Filter: {'Enabled' if cfg['volume_filter'] else 'Disabled'}
ATR Filter: {'Enabled' if cfg['atr_filter'] else 'Disabled'}
Relative Strength Filter: {'Enabled (vs NIFTY200)' if cfg['rs_filter'] else 'Disabled'}

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
    with open(summary_filename, 'w') as f: f.write(summary_content)
    
    log_columns = ['symbol', 'entry_date', 'exit_date', 'entry_price', 'pnl', 'exit_type', 'portfolio_equity_on_entry', 'portfolio_equity_on_exit', 'risk_per_share', 'initial_shares', 'initial_stop_loss']
    if not trades_df.empty:
        trades_df = trades_df.reindex(columns=log_columns)
        trades_df.to_csv(trades_filename, index=False)
    else:
        with open(trades_filename, 'w') as f: f.write(",".join(log_columns) + "\n")
    print(summary_content)
    print(f"Backtest completed in {time.time()-start_time:.2f} seconds")
    print(f"Reports saved to '{cfg['log_folder']}'")

if __name__ == "__main__":
    run_backtest(config)
