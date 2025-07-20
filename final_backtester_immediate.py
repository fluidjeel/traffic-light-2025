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
    # Possible values: 'weekly-immediate', 'monthly-immediate'
    'timeframe': 'monthly-immediate', 
    'data_folder_base': 'data/processed',
    'log_folder': 'backtest_logs',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',
    'ema_period': 30,
    'stop_loss_lookback': 5, # Note: For immediate mode, this is in CALENDAR days on the daily chart
    # --- FILTERS ---
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200',
    'regime_ma_period': 50,
    'volume_filter': True,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    'rs_filter': True,
    'rs_index_symbol': 'NIFTY200',
    'rs_period': 30
}


def parse_datetime(dt_str):
    formats = ['%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M', '%Y-%m-%d']
    for fmt in formats:
        try: return datetime.strptime(dt_str, fmt)
        except ValueError: continue
    try: return pd.to_datetime(dt_str).to_pydatetime().replace(tzinfo=None)
    except ValueError: raise ValueError(f"Time data '{dt_str}' doesn't match any expected format")


def get_consecutive_red_candles(df, current_loc):
    """Find all consecutive red candles before the green candle."""
    red_candles = []
    # The green candle is at current_loc - 1. The search for red
    # candles must start from the candle before that, at current_loc - 2.
    i = current_loc - 2 
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def run_backtest(cfg):
    start_time = time.time()
    
    base_timeframe = cfg['timeframe'].replace('-immediate', '')
    data_folder_htf = os.path.join(cfg['data_folder_base'], base_timeframe)
    data_folder_daily = os.path.join(cfg['data_folder_base'], 'daily')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(cfg['log_folder'], exist_ok=True)
    
    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
        print(f"Loaded {len(symbols)} symbols from {cfg['nifty_list_csv']}")
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}")
        return

    index_df_daily = None
    try:
        index_filename = f"{cfg['regime_index_symbol']}_INDEX_daily_with_indicators.csv"
        index_path = os.path.join(data_folder_daily, index_filename)
        index_df_daily = pd.read_csv(index_path, index_col=0, parse_dates=True)
        index_df_daily.columns = [col.lower() for col in index_df_daily.columns]
        if cfg['rs_filter']:
            index_df_daily['return'] = index_df_daily['close'].pct_change(periods=cfg['rs_period']) * 100
        print(f"Successfully loaded Daily Index data from {index_path}")
    except FileNotFoundError:
        print(f"Error: Index file not found. Disabling filters.")
        cfg['market_regime_filter'] = False
        cfg['rs_filter'] = False

    stock_data_daily = {}
    stock_data_htf = {}
    print(f"Loading and preprocessing data...")
    for symbol in symbols:
        daily_path = os.path.join(data_folder_daily, f"{symbol}_daily_with_indicators.csv")
        if os.path.exists(daily_path):
            df_d = pd.read_csv(daily_path, index_col=0, parse_dates=True)
            df_d.columns = [col.lower() for col in df_d.columns]
            df_d = df_d.loc[cfg['start_date']:cfg['end_date']]
            if cfg['volume_filter']: df_d['volume_ma'] = df_d['volume'].rolling(window=cfg['volume_ma_period']).mean()
            if cfg['rs_filter']: df_d['return'] = df_d['close'].pct_change(periods=cfg['rs_period']) * 100
            stock_data_daily[symbol] = df_d

        htf_path = os.path.join(data_folder_htf, f"{symbol}_daily_with_indicators.csv")
        if os.path.exists(htf_path):
            df_h = pd.read_csv(htf_path, index_col=0, parse_dates=True)
            df_h.columns = [col.lower() for col in df_h.columns]
            df_h = df_h.loc[cfg['start_date']:cfg['end_date']]
            if not df_h.empty:
                df_h['red_candle'] = df_h['close'] < df_h['open']
                df_h['green_candle'] = df_h['close'] > df_h['open']
                stock_data_htf[symbol] = df_h

    if not stock_data_htf: print("No valid stock data available."); return
    print(f"Successfully processed data for {len(stock_data_htf)} symbols.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    missed_trades_capital = 0
    
    all_dates = index_df_daily.loc[cfg['start_date']:cfg['end_date']].index
    
    print("Starting backtest simulation...")
    for i, date in enumerate(all_dates):
        progress_str = f"Processing {i+1}/{len(all_dates)}: {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])} | Trades: {len(portfolio['trades'])}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()
        
        exit_proceeds = 0; to_remove = []; todays_exits = [] 
        for pos_id, pos in list(portfolio['positions'].items()):
            symbol = pos['symbol']
            if symbol not in stock_data_daily or date not in stock_data_daily[symbol].index: continue
            daily_data = stock_data_daily[symbol].loc[date]
            if not pos['partial_exit'] and daily_data['high'] >= pos['target']:
                shares = pos['shares'] // 2; price = pos['target']; exit_proceeds += shares * price
                todays_exits.append({'symbol': symbol, 'entry_date': pos['entry_date'].date(), 'exit_date': date.date(), 'entry_price': pos['entry_price'], 'pnl': (price - pos['entry_price']) * shares, 'exit_type': 'Partial Profit (1:1)', **pos})
                pos['shares'] -= shares; pos['partial_exit'] = True; pos['stop_loss'] = pos['entry_price'] 
            if pos['shares'] > 0 and daily_data['low'] <= pos['stop_loss']:
                price = pos['stop_loss']; exit_proceeds += pos['shares'] * price
                # BUG FIX: Correct P&L calculation for stop-loss
                todays_exits.append({'symbol': symbol, 'entry_date': pos['entry_date'].date(), 'exit_date': date.date(), 'entry_price': pos['entry_price'], 'pnl': (price - pos['entry_price']) * pos['shares'], 'exit_type': 'Stop-Loss', **pos})
                to_remove.append(pos_id); continue
            if pos['shares'] > 0 and daily_data['close'] > pos['entry_price']:
                pos['stop_loss'] = max(pos['stop_loss'], pos['entry_price'])
                if daily_data['close'] > daily_data['open']: pos['stop_loss'] = max(pos['stop_loss'], daily_data['low'])
        for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
        portfolio['cash'] += exit_proceeds
        equity_after_exits = portfolio['cash']
        for pos in portfolio['positions'].values():
             if pos['symbol'] in stock_data_daily and date in stock_data_daily[pos['symbol']].index:
                equity_after_exits += pos['shares'] * stock_data_daily[pos['symbol']].loc[date]['close']
        portfolio['equity'] = equity_after_exits

        market_uptrend = True
        if cfg['market_regime_filter'] and date in index_df_daily.index:
            if index_df_daily.loc[date]['close'] < index_df_daily.loc[date][f"ema_{cfg['regime_ma_period']}"]: market_uptrend = False
        
        if market_uptrend:
            for symbol, df_h in stock_data_htf.items():
                if any(pos['symbol'] == symbol for pos in portfolio['positions'].values()): continue
                if symbol not in stock_data_daily: continue
                df_d = stock_data_daily[symbol]
                if date not in df_d.index: continue
                try:
                    htf_loc = df_h.index.searchsorted(date)
                    if htf_loc < 2: continue
                    prev1_h = df_h.iloc[htf_loc-1]
                    if not prev1_h['green_candle']: continue
                    red_candles_h = get_consecutive_red_candles(df_h, htf_loc)
                    if not red_candles_h: continue
                    if prev1_h['close'] < (prev1_h['high'] + prev1_h['low']) / 2: continue
                    if not (prev1_h['close'] > prev1_h[f"ema_{cfg['ema_period']}"]): continue
                    
                    entry_trigger_price = max([c['high'] for c in red_candles_h])
                    today_daily_candle = df_d.loc[date]
                    if today_daily_candle['high'] >= entry_trigger_price:
                        start_of_htf_period = df_h.index[htf_loc-1] + timedelta(days=1)
                        days_in_period = df_d.loc[start_of_htf_period:date]
                        if days_in_period.empty or days_in_period.iloc[:-1]['high'].max() < entry_trigger_price:
                            rs_ok = not cfg['rs_filter'] or (date in index_df_daily.index and df_d.loc[date, 'return'] > index_df_daily.loc[date, 'return'])
                            volume_ok = not cfg['volume_filter'] or (pd.notna(today_daily_candle['volume_ma']) and today_daily_candle['volume'] >= (today_daily_candle['volume_ma'] * cfg['volume_multiplier']))
                            if rs_ok and volume_ok:
                                entry_price = entry_trigger_price
                                loc_d = df_d.index.get_loc(date)
                                stop_loss = df_d.iloc[max(0, loc_d - cfg['stop_loss_lookback']):loc_d]['low'].min()
                                risk_per_share = entry_price - stop_loss
                                
                                # --- SANITY CHECK ---
                                if risk_per_share < (entry_price * 0.001): continue

                                equity_at_entry = portfolio['equity']
                                shares = math.floor((equity_at_entry * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)
                                if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                                    portfolio['cash'] -= shares * entry_price
                                    portfolio['positions'][f"{symbol}_{date}"] = {'symbol': symbol, 'entry_date': date, 'entry_price': entry_price, 'stop_loss': stop_loss, 'shares': shares, 'target': entry_price + risk_per_share, 'partial_exit': False, 'portfolio_equity_on_entry': equity_at_entry, 'risk_per_share': risk_per_share, 'initial_shares': shares, 'initial_stop_loss': stop_loss}
                                elif shares > 0: missed_trades_capital += 1
                except Exception: pass
        
        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if pos['symbol'] in stock_data_daily and date in stock_data_daily[pos['symbol']].index: 
                eod_equity += pos['shares'] * stock_data_daily[pos['symbol']].loc[date]['close']
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
