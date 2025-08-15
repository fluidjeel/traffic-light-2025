# daily_long_breakout.py
#
# Description:
# A realistic, portfolio-level backtester for a multi-candle breakout strategy.
#
# v29 (Adaptive Regime Logic):
# - Implemented an adaptive system that uses different parameters based on the market regime.
# - "Offensive" playbook (loose stops, higher risk) is used in "Strong Uptrend" regimes.
# - "Defensive" playbook (tight stops, lower risk) is used in "Weakening" regimes.

import pandas as pd
import os
import math
from datetime import datetime
import time
import sys
import numpy as np
import json
import pandas_ta as ta

# --- CONFIGURATION ---
config = {
    # --- General Backtest Parameters ---
    'initial_capital': 1000000, # The starting capital for the simulation.
    'start_date': '2018-01-01', # The date the backtest will begin.
    'end_date': '2025-07-31',   # The date the backtest will end.

    # --- Strategy-Specific Settings ---
    'strategy_name': 'multi_candle_breakout_adaptive', # A name for the strategy run.
    'nifty_list_csv': 'nifty500.csv', # The CSV file containing the universe of stock symbols to trade.
    'breakout_lookback_period': 2, # The number of days to look back for the breakout high/low.

    # --- Market Regime Filter ---
    'market_regime_filter': {
        'enabled': True,
        'index_symbol': 'NIFTY500-INDEX',
        'slow_ma_period': 50,
        'fast_ma_period': 20,
    },

    # --- NEW: Adaptive Parameters based on Market Regime ---
    'adaptive_regime_parameters': {
        'enabled': True,
        'strong_uptrend': { # Offensive Playbook for Bull Markets
            'risk_percents': [0.5, 0.5, 1.0],
            'high_rsi_multiplier': 8.0,
            'max_long_ema_dist_pct': 100.0
        },
        'weakening': { # Defensive Playbook for Choppy/Corrective Markets
            'risk_percents': [0.5, 0.25, 1.0],
            'high_rsi_multiplier': 4.0,
            'max_long_ema_dist_pct': 10.0
        }
    },

    # --- Realism & Risk Management ---
    'max_open_positions': 10,
    'slippage_on_entry_percent': 0.05,
    'slippage_on_exit_percent': 0.05,

    # --- Dynamic Risk Management (VIX-based) ---
    'dynamic_risk': {
        'enabled': True,
        'vix_thresholds': [15, 22],
        'risk_percents': [0.5, 0.5, 1.0] # Default/Fallback values
    },
    'risk_per_trade_percent': 2.0,
    
    # --- Entry Filters ---
    'entry_filters': {
        'use_ema_filter': True,
        'ema_period': 10,
        'use_ema_distance_filter': True,
        'min_long_ema_dist_pct': 0.0,
        'max_long_ema_dist_pct': 100.0, # Default/Fallback value
        'use_volume_filter': True,
        'min_volume_ratio': 3.0,
        'use_rsi_filter': True,
        'rsi_ranges': [[0, 40], [70, 100]],
        'min_risk_percent': 0.1,
    },

    # --- Trade Management & Trailing Stops ---
    'trade_management': {
        'use_atr': True,
        'use_breakeven': True,
        'atr_period': 14,
        'breakeven_buffer_percent': 0.1,
        'dynamic_atr': {
            'enabled': True,
            'rsi_threshold': 60,
            'low_rsi_multiplier': 3.0,
            'high_rsi_multiplier': 8.0 # Default/Fallback value
        },
    },

    # --- Data & Logging ---
    'data_folder': os.path.join('data', 'universal_processed', 'daily'),
    'log_folder': 'backtest_logs',
    'log_options': {
        'log_trades': True,
        'log_summary': True,
        'log_missed_trades': True
    },
    'vix_symbol': 'INDIAVIX',
}


# --- HELPER FUNCTIONS ---
def format_config_for_summary(cfg):
    """Formats the config dictionary into a readable string."""
    output = []
    for key, value in cfg.items():
        if isinstance(value, dict):
            output.append(f"\n[{key.replace('_', ' ').title()}]")
            for sub_key, sub_value in value.items():
                output.append(f"  {sub_key}: {sub_value}")
        else:
            output.append(f"{key}: {value}")
    return "\n".join(output)

def create_enhanced_trade_log(pos, exit_time, exit_price, exit_type):
    """Creates a detailed log entry for a closed trade."""
    base_log = pos.copy()
    pnl = (exit_price - pos['entry_price']) * pos['shares']
    initial_risk_per_share = abs(pos['entry_price'] - pos['initial_stop_loss'])
    
    if initial_risk_per_share > 0:
        mae_price = pos.get('lowest_price_since_entry', pos['entry_price'])
        mfe_price = pos.get('highest_price_since_entry', pos['entry_price'])
        mae_R = (pos['entry_price'] - mae_price) / initial_risk_per_share
        mfe_R = (mfe_price - pos['entry_price']) / initial_risk_per_share
    else:
        mae_R, mfe_R = 0, 0

    base_log.update({
        'exit_date': exit_time, 'exit_price': exit_price, 'pnl': pnl, 'exit_type': exit_type,
        'mae_R': mae_R, 'mfe_R': mfe_R
    })
    for key in ['lowest_price_since_entry', 'highest_price_since_entry', 'breakeven_triggered', 'target', 'capital_at_risk', 'last_close']:
        base_log.pop(key, None)
    return base_log

def get_dynamic_risk(vix_value, dr_cfg, current_risk_percents):
    """Determines the risk percentage based on the current VIX value and active regime."""
    if not dr_cfg['enabled'] or pd.isna(vix_value):
        return config['risk_per_trade_percent']
    
    if vix_value <= dr_cfg['vix_thresholds'][0]:
        return current_risk_percents[0]
    elif vix_value <= dr_cfg['vix_thresholds'][1]:
        return current_risk_percents[1]
    else:
        return current_risk_percents[2]

# --- MAIN PORTFOLIO SIMULATOR ---
def run_portfolio_backtest(cfg):
    """Main function to run the portfolio-level backtest."""

    print(f"--- Starting Portfolio Backtest: {cfg['strategy_name']} ---")
    
    try:
        symbols = pd.read_csv(cfg['nifty_list_csv'])['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: Symbol file not found at {cfg['nifty_list_csv']}"); return

    print(f"Loading data for {len(symbols)} symbols...")
    all_data = {}
    master_dates = set()
    
    regime_cfg = cfg.get('market_regime_filter', {})
    symbols_to_load = symbols + [cfg['vix_symbol']]
    if regime_cfg.get('enabled'):
        symbols_to_load.append(regime_cfg['index_symbol'])

    for symbol in list(set(symbols_to_load)):
        file_path = os.path.join(cfg['data_folder'], f"{symbol}_daily_with_indicators.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
            df.rename(columns=lambda x: x.lower(), inplace=True)
            
            df_filtered = df.loc[cfg['start_date']:(pd.to_datetime(cfg['end_date']) + pd.Timedelta(days=1))]
            if not df_filtered.empty:
                all_data[symbol] = df_filtered
                if symbol not in [cfg['vix_symbol'], regime_cfg.get('index_symbol')]:
                    master_dates.update(df_filtered.index)
    
    vix_data = all_data.get(cfg['vix_symbol'])
    regime_data = all_data.get(regime_cfg.get('index_symbol')) if regime_cfg.get('enabled') else None

    master_dates = sorted(list(master_dates))
    print(f"Data loading complete. Running simulation from {master_dates[0].date()} to {master_dates[-1].date()}.")

    portfolio = {'cash': cfg['initial_capital'], 'equity': cfg['initial_capital'], 'positions': {}, 'trades': [], 'daily_values': []}
    missed_trades_log = []
    watchlist = {}
    
    tm_cfg = cfg['trade_management']
    ef_cfg = cfg['entry_filters']
    dr_cfg = cfg['dynamic_risk']
    adaptive_cfg = cfg.get('adaptive_regime_parameters', {})
    
    eod_equity_yesterday = cfg['initial_capital']

    for i, date in enumerate(master_dates):
        # --- 1. DETERMINE CURRENT DAY'S REGIME AND PARAMETERS ---
        regime_status = "N/A"
        scan_for_setups = True
        
        # Set default parameters (Offensive Playbook)
        active_params = adaptive_cfg.get('strong_uptrend', {})
        
        if regime_cfg.get('enabled') and regime_data is not None:
            if date in regime_data.index:
                regime_candle = regime_data.loc[date]
                slow_ma_col = f"sma_{regime_cfg['slow_ma_period']}"
                fast_ma_col = f"sma_{regime_cfg['fast_ma_period']}"
                
                slow_ma = regime_candle.get(slow_ma_col)
                fast_ma = regime_candle.get(fast_ma_col)
                close = regime_candle['close']

                if pd.notna(slow_ma) and pd.notna(fast_ma):
                    if close < slow_ma:
                        regime_status = "Downtrend"
                        scan_for_setups = False
                    elif close < fast_ma:
                        regime_status = "Weakening"
                        # Switch to Defensive Playbook
                        if adaptive_cfg.get('enabled'):
                            active_params = adaptive_cfg.get('weakening', {})
                    else:
                        regime_status = "Strong Uptrend"
                else:
                    scan_for_setups = False
            else:
                scan_for_setups = False
        
        # Unpack the active parameters for today's logic
        current_risk_percents = active_params.get('risk_percents', dr_cfg['risk_percents'])
        current_high_rsi_mult = active_params.get('high_rsi_multiplier', tm_cfg['dynamic_atr']['high_rsi_multiplier'])
        current_max_ema_dist = active_params.get('max_long_ema_dist_pct', ef_cfg['max_long_ema_dist_pct'])

        current_vix = vix_data.loc[date]['close'] if vix_data is not None and date in vix_data.index else np.nan
        current_risk_percent = get_dynamic_risk(current_vix, dr_cfg, current_risk_percents)

        progress_str = f"Processing {date.date()} | Equity: {portfolio['equity']:,.0f} | Regime: {regime_status} | Open Pos: {len(portfolio['positions'])}"
        sys.stdout.write(f"\r{progress_str.ljust(110)}"); sys.stdout.flush()

        # --- 2. EOD Trailing Stop Update (Using T-1 Data) ---
        for symbol, pos in portfolio['positions'].items():
            if symbol in all_data and date in all_data[symbol].index:
                today_candle = all_data[symbol].loc[date]
                if not pos['breakeven_triggered'] and tm_cfg.get('use_breakeven'):
                    if today_candle['close'] > pos['entry_price']:
                        buffer_amount = pos['entry_price'] * (tm_cfg['breakeven_buffer_percent'] / 100)
                        pos['stop_loss'] = max(pos['stop_loss'], pos['entry_price'] + buffer_amount)
                        pos['breakeven_triggered'] = True

                if date > pos['entry_date'] and tm_cfg.get('use_atr'):
                    prev_day_index = all_data[symbol].index.get_loc(date) - 1
                    if prev_day_index >= 0:
                        prev_day_data = all_data[symbol].iloc[prev_day_index]
                        atr_val = prev_day_data.get(f"atr_{tm_cfg['atr_period']}", 0)
                        
                        atr_mult = current_high_rsi_mult # Use the adaptive multiplier
                        dyn_atr_cfg = tm_cfg.get('dynamic_atr')
                        if dyn_atr_cfg and dyn_atr_cfg.get('enabled'):
                            rsi_at_entry = pos.get('rsi_at_entry', 60)
                            if rsi_at_entry < dyn_atr_cfg['rsi_threshold']:
                                atr_mult = dyn_atr_cfg['low_rsi_multiplier']
                        
                        primary_atr_trail_value = pos['highest_price_since_entry'] - (atr_val * atr_mult)
                        pos['stop_loss'] = max(pos['stop_loss'], primary_atr_trail_value)
        
        # --- 3. Check for Stop-Loss Hits & Update Watermarks ---
        positions_to_close = []
        for symbol, pos in list(portfolio['positions'].items()):
            if symbol not in all_data or date not in all_data[symbol].index: continue
            
            today_candle = all_data[symbol].loc[date]
            
            if today_candle['open'] <= pos['stop_loss']:
                exit_price = today_candle['open'] * (1 - cfg.get('slippage_on_exit_percent', 0) / 100)
                exit_trade = create_enhanced_trade_log(pos, date, exit_price, 'Stop-Loss (Gap)')
                portfolio['trades'].append(exit_trade)
                portfolio['cash'] += exit_trade['shares'] * exit_trade['exit_price']
                positions_to_close.append(symbol)
                continue

            if today_candle['low'] <= pos['stop_loss']:
                exit_price = pos['stop_loss'] * (1 - cfg.get('slippage_on_exit_percent', 0) / 100)
                exit_trade = create_enhanced_trade_log(pos, date, exit_price, 'Stop-Loss')
                portfolio['trades'].append(exit_trade)
                portfolio['cash'] += exit_trade['shares'] * exit_trade['exit_price']
                positions_to_close.append(symbol)
                continue

            pos['lowest_price_since_entry'] = min(pos.get('lowest_price_since_entry', pos['entry_price']), today_candle['low'])
            pos['highest_price_since_entry'] = max(pos.get('highest_price_since_entry', pos['entry_price']), today_candle['high'])

        for symbol in positions_to_close: portfolio['positions'].pop(symbol, None)

        # --- 4. Check Watchlist for New Entries ---
        todays_watchlist = watchlist.get(date, {})
        
        for symbol, order in todays_watchlist.items():
            rejection_reason = None
            
            if len(portfolio['positions']) >= cfg['max_open_positions']:
                rejection_reason = "Entry Rejected: Max open positions reached"

            if rejection_reason:
                if cfg['log_options'].get('log_missed_trades'):
                    missed_trades_log.append({'date': date, 'symbol': symbol, 'stage': 'Entry', 'reason': rejection_reason, 'vix': current_vix})
                continue

            if symbol not in all_data or date not in all_data[symbol].index:
                rejection_reason = "Entry Rejected: No data for entry day"
            else:
                today_candle = all_data[symbol].loc[date]
                if today_candle['high'] < order['trigger_price']:
                    rejection_reason = "Entry Not Triggered: High < Trigger Price"
                else:
                    entry_price_base, stop_loss_price = order['trigger_price'], order['stop_loss_price']
                    entry_price = entry_price_base * (1 + cfg.get('slippage_on_entry_percent', 0) / 100)
                    
                    effective_stop = stop_loss_price * (1 - cfg.get('slippage_on_exit_percent', 0) / 100)
                    risk_per_share = entry_price - effective_stop

                    if risk_per_share <= (entry_price * (ef_cfg['min_risk_percent'] / 100)):
                        rejection_reason = "Entry Rejected: Risk per share too low"
                    else:
                        capital_at_risk = eod_equity_yesterday * (current_risk_percent / 100)
                        shares = math.floor(capital_at_risk / risk_per_share) if risk_per_share > 0 else 0
                        if shares == 0 or (shares * entry_price) > portfolio['cash']:
                            rejection_reason = "Entry Rejected: Insufficient capital"

            if rejection_reason:
                if cfg['log_options'].get('log_missed_trades'):
                    missed_trades_log.append({'date': date, 'symbol': symbol, 'stage': 'Entry', 'reason': rejection_reason, 'vix': current_vix})
                continue
            
            # --- Execute New Trade ---
            capital_at_risk = eod_equity_yesterday * (current_risk_percent / 100)
            entry_price_base, stop_loss_price = order['trigger_price'], order['stop_loss_price']
            entry_price = entry_price_base * (1 + cfg.get('slippage_on_entry_percent', 0) / 100)
            effective_stop = stop_loss_price * (1 - cfg.get('slippage_on_exit_percent', 0) / 100)
            risk_per_share = entry_price - effective_stop
            shares = math.floor(capital_at_risk / risk_per_share) if risk_per_share > 0 else 0
            
            portfolio['cash'] -= shares * entry_price
            
            setup_candle_data = all_data[symbol].loc[order['setup_candle_date']]
            ema_col = f"ema_{ef_cfg['ema_period']}"
            ema_val = setup_candle_data.get(ema_col, np.nan)
            dist_pct = ((setup_candle_data['close'] / ema_val - 1) * 100) if pd.notna(ema_val) and ema_val > 0 else np.nan
            
            entry_day_candle = all_data[symbol].loc[date]
            initial_high = max(entry_price, entry_day_candle['high'])
            initial_low = min(entry_price, entry_day_candle['low'])

            portfolio['positions'][symbol] = {
                'symbol': symbol, 'direction': 'long', 'entry_date': date, 'entry_price': entry_price, 'shares': shares,
                'stop_loss': stop_loss_price, 'initial_stop_loss': stop_loss_price, 'target': np.inf,
                'setup_candle_date': order['setup_candle_date'],
                'lowest_price_since_entry': initial_low, 
                'highest_price_since_entry': initial_high, 
                'breakeven_triggered': False,
                'capital_at_risk': capital_at_risk, 
                'vix_at_entry': current_vix,
                'risk_pct_at_entry': current_risk_percent,
                'rsi_at_entry': setup_candle_data.get('rsi_14', np.nan),
                'volume_ratio_at_entry': setup_candle_data.get('volume_ratio', np.nan),
                'close_to_ema_dist_pct_at_entry': dist_pct,
                'last_close': entry_day_candle['close'],
            }

        # --- 5. EOD Equity Update ---
        eod_equity = portfolio['cash']
        for symbol, pos in portfolio['positions'].items():
            if symbol in all_data and date in all_data[symbol].index:
                pos['last_close'] = all_data[symbol].loc[date]['close']
            eod_equity += pos['shares'] * pos['last_close']

        portfolio['equity'] = eod_equity
        eod_equity_yesterday = eod_equity
        portfolio['daily_values'].append({'date': date, 'equity': eod_equity})

        # --- 6. Scan for New Setups for Next Day ---
        if i + 1 < len(master_dates) and scan_for_setups:
            next_day = master_dates[i+1]
            watchlist[next_day] = {}
            for symbol in symbols:
                if symbol in portfolio['positions'] or symbol not in all_data or date not in all_data[symbol].index: continue
                
                setup_candle = all_data[symbol].loc[date]
                
                is_valid_setup = True
                rejection_reason = None

                if ef_cfg.get('use_volume_filter') and setup_candle.get('volume_ratio', 0) < ef_cfg['min_volume_ratio']:
                    is_valid_setup = False
                    rejection_reason = "Setup Rejected: Low Volume Ratio"

                if is_valid_setup:
                    ema_val = setup_candle.get(f"ema_{ef_cfg['ema_period']}", np.nan)
                    if ef_cfg.get('use_ema_filter') and (pd.isna(ema_val) or setup_candle['close'] <= ema_val):
                        is_valid_setup = False
                        rejection_reason = "Setup Rejected: Close Not Above EMA"

                if is_valid_setup:
                    if ef_cfg.get('use_ema_distance_filter'):
                        ema_val = setup_candle.get(f"ema_{ef_cfg['ema_period']}", np.nan) 
                        if pd.notna(ema_val) and ema_val > 0:
                            dist_pct = (setup_candle['close'] / ema_val - 1) * 100
                            if not (ef_cfg['min_long_ema_dist_pct'] <= dist_pct <= current_max_ema_dist): # Use adaptive value
                                is_valid_setup = False
                                rejection_reason = "Setup Rejected: EMA Distance Out of Range"
                        else:
                            is_valid_setup = False
                            rejection_reason = "Setup Rejected: Invalid EMA for Distance Calc"
                
                if is_valid_setup and ef_cfg.get('use_rsi_filter'):
                    rsi_val = setup_candle.get('rsi_14', 50)
                    is_in_any_range = False
                    for rsi_range in ef_cfg.get('rsi_ranges', []):
                        if rsi_range[0] <= rsi_val <= rsi_range[1]:
                            is_in_any_range = True
                            break
                    if not is_in_any_range:
                        is_valid_setup = False
                        rejection_reason = "Setup Rejected: RSI Out of Range"

                if is_valid_setup:
                    lookback_period = cfg.get('breakout_lookback_period', 1)
                    symbol_data = all_data[symbol]
                    
                    try:
                        current_day_index = symbol_data.index.get_loc(date)
                        
                        if current_day_index >= lookback_period - 1:
                            start_index = current_day_index - (lookback_period - 1)
                            lookback_candles = symbol_data.iloc[start_index : current_day_index + 1]
                            
                            trigger_price = lookback_candles['high'].max()
                            stop_loss_price = lookback_candles['low'].min()

                            watchlist[next_day][symbol] = {'direction': 'long', 'trigger_price': trigger_price, 'stop_loss_price': stop_loss_price, 'setup_candle_date': date}
                        else:
                            rejection_reason = "Setup Rejected: Not enough data for lookback"
                    except KeyError:
                        rejection_reason = "Setup Rejected: Date not found in symbol data"

                if not is_valid_setup and cfg['log_options'].get('log_missed_trades'):
                    missed_trades_log.append({'date': date, 'symbol': symbol, 'stage': 'Setup', 'reason': rejection_reason, 'vix': current_vix})


    # --- Final Report Generation ---
    print("\n\n--- AGGREGATING RESULTS ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_log_folder = os.path.join(cfg['log_folder'], cfg['strategy_name'], timestamp)
    os.makedirs(strategy_log_folder, exist_ok=True)
    
    if cfg['log_options']['log_summary']:
        equity_df = pd.DataFrame(portfolio['daily_values']).set_index('date')
        trades_df = pd.DataFrame(portfolio['trades'])
        
        cagr, max_drawdown, sharpe_ratio = 0, 0, 0
        if not equity_df.empty and len(equity_df) > 1:
            years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
            cagr = ((portfolio['equity'] / cfg['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 else 0
            peak = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - peak) / peak
            max_drawdown = abs(drawdown.min()) * 100
            
            daily_returns = equity_df['equity'].pct_change().dropna()
            if not daily_returns.empty and daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        total_trades = len(trades_df)
        profit_factor, payoff_ratio, avg_holding_period = 0, 0, 'N/A'
        if total_trades > 0:
            winners = trades_df[trades_df['pnl'] > 0]
            losers = trades_df[trades_df['pnl'] <= 0]
            
            gross_profit = winners['pnl'].sum()
            gross_loss = abs(losers['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

            avg_win = winners['pnl'].mean() if not winners.empty else 0
            avg_loss = abs(losers['pnl'].mean()) if not losers.empty else 0
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf

            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            avg_holding_period_days = (trades_df['exit_date'] - trades_df['entry_date']).mean().days
            avg_holding_period = f"{avg_holding_period_days} days"

        
        config_str = format_config_for_summary(cfg)
        summary_text = f"""
BACKTEST PARAMETERS
======================================================
{config_str}

PERFORMANCE SUMMARY
======================================================
[Key Metrics]
  Final Equity: {portfolio['equity']:,.2f}
  Net PnL: {trades_df['pnl'].sum():,.2f}
  CAGR: {cagr:.2f}%
  Max Drawdown: {max_drawdown:.2f}%
  Sharpe Ratio: {sharpe_ratio:.2f}

[Trade Stats]
  Total Trades: {total_trades}
  Win Rate: {(len(trades_df[trades_df['pnl'] > 0]) / total_trades * 100) if total_trades > 0 else 0:.2f}%
  Profit Factor: {profit_factor:.2f}
  Payoff Ratio (Avg Win/Loss): {payoff_ratio:.2f}
  Avg. Holding Period: {avg_holding_period}

[System Activity]
  Total Setups Rejected: {len([m for m in missed_trades_log if m['stage'] == 'Setup'])}
  Total Entries Missed: {len([m for m in missed_trades_log if m['stage'] == 'Entry'])}
"""
        print(summary_text)
        summary_path = os.path.join(strategy_log_folder, f"{timestamp}_summary.txt")
        with open(summary_path, 'w') as f: f.write(summary_text)
        print(f"Universe summary report saved to '{summary_path}'")

    if cfg['log_options']['log_trades'] and not trades_df.empty:
        trades_path = os.path.join(strategy_log_folder, f"{timestamp}_all_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Aggregated trade log saved to '{trades_path}'")
    
    if cfg['log_options'].get('log_missed_trades', False) and missed_trades_log:
        missed_trades_df = pd.DataFrame(missed_trades_log)
        missed_trades_path = os.path.join(strategy_log_folder, f"{timestamp}_missed_trades.csv")
        missed_trades_df.to_csv(missed_trades_path, index=False)
        print(f"Missed trades log saved to '{missed_trades_path}'")

if __name__ == "__main__":
    run_portfolio_backtest(config)