import pandas as pd
import os
from datetime import datetime, timedelta

# Configuration
CONFIG = {
    "equity": 1000000,
    "risk_per_trade_pct": 0.02,
    "max_portfolio_risk_pct": 0.10,
    "weekly_data_dir": "data/processed/weekly/",
    "daily_data_dir": "data/processed/daily/",
    "intraday_data_dir": "historical_data_15min/",
    "log_dir": "backtest_logs/htf_dynamic_positioning/",
    "volume_projection_checkpoints": ["11:30", "14:15", "14:30", "14:45", "15:00"],
    "volume_threshold_pct": 85.0
}

os.makedirs(CONFIG["log_dir"], exist_ok=True)

class PortfolioManager:
    def __init__(self, equity, risk_per_trade_pct, max_portfolio_risk_pct):
        self.equity = equity
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.open_trades = []

    def total_open_risk(self):
        return sum([t['risk_per_share'] * t['shares'] for t in self.open_trades])

    def can_take_trade(self, risk_per_share):
        risk_per_trade = self.equity * self.risk_per_trade_pct
        if risk_per_share <= 0:
            return 0, False
        shares = int(risk_per_trade // risk_per_share)
        projected_risk = self.total_open_risk() + (shares * risk_per_share)
        if projected_risk > self.equity * self.max_portfolio_risk_pct:
            return 0, False
        return shares, True

    def add_trade(self, symbol, entry_date, entry_price, stop_price, shares):
        trade = {
            'symbol': symbol,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'shares': shares,
            'risk_per_share': entry_price - stop_price
        }
        self.open_trades.append(trade)

def load_weekly_daily_data(symbol):
    weekly_path = os.path.join(CONFIG['weekly_data_dir'], f"{symbol}_weekly_with_indicators.csv")
    daily_path = os.path.join(CONFIG['daily_data_dir'], f"{symbol}_daily_with_indicators.csv")
    if not os.path.exists(weekly_path) or not os.path.exists(daily_path):
        print(f"Missing file for {symbol} - Skipping")
        return None, None
    weekly_df = pd.read_csv(weekly_path, parse_dates=['datetime'])
    daily_df = pd.read_csv(daily_path, parse_dates=['datetime'])
    return weekly_df, daily_df

def load_intraday_data(symbol):
    path = os.path.join(CONFIG['intraday_data_dir'], f"{symbol}_intraday_15min.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['datetime'])
    return df

def get_all_symbols():
    return [f.split("_weekly_with_indicators.csv")[0] for f in os.listdir(CONFIG['weekly_data_dir']) if f.endswith("_weekly_with_indicators.csv")]

def write_logs(logs, filename):
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(CONFIG['log_dir'], filename), index=False)

def passes_volume_projection(symbol, entry_datetime, avg_volume):
    intraday_df = load_intraday_data(symbol)
    if intraday_df is None:
        return True, "NO_INTRADAY_DATA"
    date_str = entry_datetime.strftime('%Y-%m-%d')
    daily_data = intraday_df[intraday_df['datetime'].dt.strftime('%Y-%m-%d') == date_str].copy()
    if daily_data.empty:
        return True, "NO_INTRADAY_MATCH"
    last_checkpoint = max([pd.to_datetime(f"{date_str} {t}") for t in CONFIG['volume_projection_checkpoints'] if pd.to_datetime(f"{date_str} {t}") <= entry_datetime], default=None)
    if last_checkpoint is None:
        return True, "BEFORE_CHECKPOINTS"
    up_to_now = daily_data[daily_data['datetime'] <= last_checkpoint]
    cum_vol = up_to_now['volume'].sum()
    threshold_vol = avg_volume * (CONFIG['volume_threshold_pct'] / 100)
    if cum_vol < threshold_vol:
        return False, f"REJECTED_VOL {cum_vol}/{avg_volume:.0f}"
    return True, "PASS"

def find_htf_trades(symbol, weekly_df, daily_df, portfolio):
    logs = []
    if len(weekly_df) < 5:
        return logs
    latest_week = weekly_df.iloc[-2]
    prior_weeks = weekly_df.iloc[:-2].copy()
    red_weeks = prior_weeks[prior_weeks['close'] < prior_weeks['open']]

    if latest_week['close'] <= latest_week['open']:
        return logs
    if latest_week['close'] <= latest_week['ema_30']:
        return logs
    if latest_week['close'] > (latest_week['high'] + latest_week['low']) / 2:
        return logs
    if red_weeks.empty:
        return logs

    trigger_price = red_weeks['high'].max()
    week_start = latest_week['datetime'] + timedelta(days=1)
    week_end = week_start + timedelta(days=6)
    sniper_df = daily_df[(daily_df['datetime'] >= week_start) & (daily_df['datetime'] <= week_end)].copy()
    sniper_df = sniper_df[sniper_df['high'] >= trigger_price]
    if sniper_df.empty:
        return logs

    first_breakout = sniper_df.iloc[0]
    entry_price = trigger_price
    stop_price = first_breakout['low']
    entry_date = first_breakout['datetime']
    avg_volume = first_breakout.get('volume_20_sma', 0)

    passes_vol, vol_reason = passes_volume_projection(symbol, entry_date, avg_volume)
    if not passes_vol:
        logs.append({
            'symbol': symbol,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'shares': 0,
            'status': vol_reason
        })
        return logs

    risk_per_share = entry_price - stop_price
    shares, can_trade = portfolio.can_take_trade(risk_per_share)

    if can_trade and shares > 0:
        portfolio.add_trade(symbol, entry_date, entry_price, stop_price, shares)
        logs.append({
            'symbol': symbol,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'shares': shares,
            'status': 'TRADE_ENTERED'
        })
    else:
        logs.append({
            'symbol': symbol,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'shares': 0,
            'status': 'SKIPPED_MAX_RISK'
        })
    return logs

def main():
    symbols = get_all_symbols()
    portfolio = PortfolioManager(CONFIG['equity'], CONFIG['risk_per_trade_pct'], CONFIG['max_portfolio_risk_pct'])
    all_logs = []

    for symbol in symbols:
        weekly_df, daily_df = load_weekly_daily_data(symbol)
        if weekly_df is None or daily_df is None:
            continue
        logs = find_htf_trades(symbol, weekly_df, daily_df, portfolio)
        all_logs.extend(logs)

    write_logs(all_logs, "all_trades.csv")
    print("HTF simulation with volume conviction filter complete.")

if __name__ == "__main__":
    main()
