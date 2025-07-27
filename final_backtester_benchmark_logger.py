# final_backtester_benchmark_logger.py
#
# Description:
# This script runs the original, flawed strategy with lookahead bias.
# Its sole purpose is to generate the "Golden Benchmark" log file, which
# contains every "perfect" trade the system could find with future knowledge.
# It also tracks the hypothetical PnL of every setup.

import pandas as pd
import os
import math
from datetime import datetime
import time
import sys

# --- CONFIGURATION ---
config = {
    'initial_capital': 1000000,
    'risk_per_trade_percent': 4.0,
    'timeframe': 'daily',
    'data_folder_base': 'data/processed',
    'log_folder': 'backtest_logs',
    'start_date': '2020-01-01',
    'end_date': '2025-07-16',
    'nifty_list_csv': 'nifty200.csv',

    # --- STRATEGY PARAMETERS ---
    'ema_period': 30,
    'stop_loss_lookback': 5,

    # --- FILTERS (Lookahead Bias Preserved) ---
    'market_regime_filter': True,
    'regime_index_symbol': 'NIFTY200_INDEX',
    'regime_ma_period': 50,
    'volume_filter': True,
    'volume_ma_period': 20,
    'volume_multiplier': 1.3,
    'rs_filter': True,
    'rs_index_symbol': 'NIFTY200_INDEX',
    'rs_period': 30,

    # --- LOGGING ---
    'log_all_setups': True
}


def get_consecutive_red_candles(df, current_loc):
    red_candles = []
    i = current_loc - 2
    while i >= 0 and df.iloc[i]['red_candle']:
        red_candles.append(df.iloc[i])
        i -= 1
    return red_candles

def simulate_trade_outcome(symbol, entry_date, entry_price, stop_loss, daily_data):
    """
    Simulates the lifecycle of a single trade to find its hypothetical outcome.
    """
    df = daily_data[symbol]
    target_price = entry_price + (entry_price - stop_loss)
    current_stop = stop_loss
    partial_exit_pnl = 0
    final_pnl = 0
    leg1_sold = False
    exit_date = None

    trade_dates = df.loc[entry_date:].index[1:]

    for date in trade_dates:
        if date not in df.index: continue
        candle = df.loc[date]

        if not leg1_sold and candle['high'] >= target_price:
            partial_exit_pnl = target_price - entry_price
            leg1_sold = True
            current_stop = entry_price

        if candle['low'] <= current_stop:
            final_pnl = current_stop - entry_price
            exit_date = date
            break

        if candle['close'] > entry_price:
            current_stop = max(current_stop, entry_price)
            if candle['green_candle']:
                current_stop = max(current_stop, candle['low'])

    if exit_date is None:
        exit_date = df.index[-1]
        final_pnl = df.iloc[-1]['close'] - entry_price

    total_pnl = (partial_exit_pnl * 0.5) + (final_pnl * 0.5) if leg1_sold else final_pnl
    return exit_date, total_pnl


def run_backtest(cfg):
    start_time = time.time()

    data_folder = os.path.join(cfg['data_folder_base'], cfg['timeframe'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(cfg['log_folder'], exist_ok=True)

    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}")
        return

    print("Loading all daily data into memory...")
    daily_data = {}
    for symbol in symbols + [cfg['regime_index_symbol']]:
        try:
            file_path = os.path.join(data_folder, f"{symbol}_{cfg['timeframe']}_with_indicators.csv")
            df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
            df['red_candle'] = df['close'] < df['open']
            df['green_candle'] = df['close'] > df['open']
            daily_data[symbol] = df
        except FileNotFoundError:
            print(f"Warning: Data for {symbol} not found. Skipping.")
    print("Data loading complete.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    all_setups_log = []

    master_dates = daily_data.get(cfg['regime_index_symbol'], pd.DataFrame()).loc[cfg['start_date']:cfg['end_date']].index

    print("Starting Benchmark backtest simulation...")
    for date in master_dates:
        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Positions: {len(portfolio['positions'])}"
        sys.stdout.write(f"\r{progress_str.ljust(100)}"); sys.stdout.flush()

        # EOD Exit Logic
        exit_proceeds = 0; to_remove = []
        for pos_id, pos in list(portfolio['positions'].items()):
            symbol = pos['symbol']
            if symbol not in daily_data or date not in daily_data[symbol].index: continue
            candle = daily_data[symbol].loc[date]
            if not pos['partial_exit'] and candle['high'] >= pos['target']:
                shares = pos['shares'] // 2; price = pos['target']; exit_proceeds += shares * price
                portfolio['trades'].append({'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date, 'pnl': (price - pos['entry_price']) * shares, 'exit_type': 'Partial Profit (1:1)', **pos})
                pos['shares'] -= shares; pos['partial_exit'] = True; pos['stop_loss'] = pos['entry_price']
            if pos['shares'] > 0 and candle['low'] <= pos['stop_loss']:
                price = pos['stop_loss']; exit_proceeds += pos['shares'] * price
                portfolio['trades'].append({'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date, 'pnl': (price - pos['entry_price']) * pos['shares'], 'exit_type': 'Stop-Loss', **pos})
                to_remove.append(pos_id)
        for pos_id in to_remove: portfolio['positions'].pop(pos_id, None)
        portfolio['cash'] += exit_proceeds

        equity_after_exits = portfolio['cash']
        for pos in portfolio['positions'].values():
            if pos['symbol'] in daily_data and date in daily_data[pos['symbol']].index:
                equity_after_exits += pos['shares'] * daily_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = equity_after_exits

        # EOD Entry Logic
        if cfg['market_regime_filter'] and date in daily_data[cfg['regime_index_symbol']].index:
            if daily_data[cfg['regime_index_symbol']].loc[date]['close'] < daily_data[cfg['regime_index_symbol']].loc[date][f"ema_{cfg['regime_ma_period']}"]:
                continue

        for symbol in symbols:
            if symbol not in daily_data: continue
            df = daily_data[symbol]
            if any(p['symbol'] == symbol for p in portfolio['positions'].values()): continue

            try:
                loc = df.index.get_loc(date)
                if loc < 2: continue

                prev1 = df.iloc[loc-1]
                if not prev1['green_candle']: continue
                red_candles = get_consecutive_red_candles(df, loc)
                if not red_candles: continue

                trigger_price = max([c['high'] for c in red_candles] + [prev1['high']])
                today_candle = df.iloc[loc]

                if today_candle['high'] >= trigger_price and today_candle['open'] < trigger_price:
                    # --- LOOKAHEAD BIAS FILTERS ---
                    volume_ok = not cfg['volume_filter'] or (today_candle['volume'] > today_candle[f"volume_{cfg['volume_ma_period']}_sma"] * cfg['volume_multiplier'])
                    rs_ok = not cfg['rs_filter'] or (today_candle[f"return_{cfg['rs_period']}"] > daily_data[cfg['rs_index_symbol']].loc[date][f"return_{cfg['rs_period']}"])

                    log_entry = {
                        'symbol': symbol, 'setup_date': date, 'trigger_price': trigger_price,
                        'status': 'FILTERED_OUT', 'volume_ok': volume_ok, 'rs_ok': rs_ok
                    }

                    if volume_ok and rs_ok:
                        entry_price = trigger_price
                        stop_loss = df.iloc[max(0, loc - cfg['stop_loss_lookback']):loc]['low'].min()
                        if pd.isna(stop_loss):
                            stop_loss = df.iloc[loc - 1]['low']

                        risk_per_share = entry_price - stop_loss
                        if risk_per_share <= 0: continue

                        shares = math.floor((portfolio['equity'] * (cfg['risk_per_trade_percent'] / 100)) / risk_per_share)

                        if shares > 0 and (shares * entry_price) <= portfolio['cash']:
                            log_entry['status'] = 'FILLED'
                            portfolio['cash'] -= shares * entry_price
                            portfolio['positions'][f"{symbol}_{date}"] = {
                                'symbol': symbol, 'entry_date': date, 'entry_price': entry_price,
                                'stop_loss': stop_loss, 'shares': shares, 'target': entry_price + risk_per_share,
                                'partial_exit': False
                            }
                        elif shares > 0:
                            log_entry['status'] = 'MISSED_CAPITAL'

                    if cfg['log_all_setups']:
                        all_setups_log.append(log_entry)

            except (KeyError, IndexError):
                continue

        # EOD Trailing Stop Logic
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                candle = daily_data[pos['symbol']].loc[date]
                if candle['close'] > pos['entry_price']:
                    pos['stop_loss'] = max(pos['stop_loss'], pos['entry_price'])
                    if candle['green_candle']:
                        pos['stop_loss'] = max(pos['stop_loss'], candle['low'])

        eod_equity = portfolio['cash']
        for pos in portfolio['positions'].values():
            if date in daily_data.get(pos['symbol'], pd.DataFrame()).index:
                eod_equity += pos['shares'] * daily_data[pos['symbol']].loc[date]['close']
        portfolio['equity'] = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

    # --- FINAL METRICS AND REPORTING ---
    print("\n--- BACKTEST COMPLETE ---")

    if cfg['log_all_setups']:
        print("Simulating outcomes for missed trades...")
        for log_entry in all_setups_log:
            if log_entry['status'] in ['MISSED_CAPITAL', 'FILTERED_OUT']:
                df = daily_data[log_entry['symbol']]
                loc = df.index.get_loc(log_entry['setup_date'])
                stop_loss = df.iloc[max(0, loc - cfg['stop_loss_lookback']):loc]['low'].min()
                if pd.isna(stop_loss):
                    stop_loss = df.iloc[loc - 1]['low']
                exit_date, pnl = simulate_trade_outcome(log_entry['symbol'], log_entry['setup_date'], log_entry['trigger_price'], stop_loss, daily_data)
                log_entry['hypothetical_exit_date'] = exit_date
                log_entry['hypothetical_pnl_per_share'] = pnl

    final_equity = portfolio['equity']
    net_pnl = final_equity - config['initial_capital']
    equity_df = pd.DataFrame(portfolio['daily_values'])
    equity_df.set_index('date', inplace=True)

    if not equity_df.empty:
        years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = ((final_equity / config['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100
    else:
        cagr, max_drawdown = 0, 0

    trades_df = pd.DataFrame(portfolio['trades'])
    total_trades, win_rate, profit_factor = 0, 0, 0
    if not trades_df.empty:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        total_trades = len(trades_df)
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    summary_filename = os.path.join(cfg['log_folder'], f"{timestamp}_summary_benchmark.txt")
    trades_filename = os.path.join(cfg['log_folder'], f"{timestamp}_trades_detail_benchmark.csv")
    all_setups_filename = os.path.join(cfg['log_folder'], f"{timestamp}_all_setups_log_benchmark.csv")

    with open(summary_filename, 'w') as f:
        f.write("BENCHMARK SUMMARY\n")
        f.write(f"CAGR: {cagr:.2f}%\n")
        f.write(f"Max Drawdown: {max_drawdown:.2f}%\n")
        f.write(f"Profit Factor: {profit_factor:.2f}\n")
        f.write(f"Total Trades: {total_trades}\n")

    if not trades_df.empty:
        trades_df.to_csv(trades_filename, index=False)
    if all_setups_log:
        pd.DataFrame(all_setups_log).to_csv(all_setups_filename, index=False)

    print(f"\nResults saved to {cfg['log_folder']}")


if __name__ == "__main__":
    run_backtest(config)
