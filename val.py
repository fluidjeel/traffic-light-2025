import pandas as pd
import os
import numpy as np
from datetime import datetime

class BreakoutStrategyValidator:
    def __init__(self, config, backtest_log_folder):
        self.config = config
        self.backtest_log_folder = backtest_log_folder
        self.data_folder = config['data_folder']
        self.vix_symbol = config['vix_symbol']
        
    def load_trade_data(self):
        """Load all trade logs with robust column checking"""
        trade_files = [f for f in os.listdir(self.backtest_log_folder) 
                      if f.endswith('_all_trades.csv')]
        
        all_trades = []
        for file in trade_files:
            try:
                df = pd.read_csv(os.path.join(self.backtest_log_folder, file))
                
                # Check for required columns
                required_cols = [
                    'entry_date', 'exit_date', 'entry_price', 
                    'initial_stop_loss', 'shares', 'symbol',
                    'exit_price', 'exit_type'
                ]
                
                for col in required_cols:
                    if col not in df.columns:
                        raise ValueError(f"Missing required column: {col}")
                
                # Handle optional columns
                optional_cols = {
                    'capital_at_risk': lambda x: x['entry_price'] * x['shares'] * 0.02,  # Default 2% risk
                    'stop_loss_price': lambda x: x['initial_stop_loss'],
                    'setup_candle_date': lambda x: x['entry_date'],
                    'rsi_at_entry': 50
                }
                
                for col, default in optional_cols.items():
                    if col not in df.columns:
                        if callable(default):
                            df[col] = default(df)
                        else:
                            df[col] = default
                
                df['entry_date'] = pd.to_datetime(df['entry_date'])
                df['exit_date'] = pd.to_datetime(df['exit_date'])
                all_trades.append(df)
                
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
                
        if not all_trades:
            raise ValueError("No valid trade files found")
            
        return pd.concat(all_trades).reset_index(drop=True)

    def load_symbol_data(self, symbol):
        """Load symbol data with error handling"""
        try:
            file_path = os.path.join(self.data_folder, f"{symbol}_daily_with_indicators.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found for {symbol}")
                
            df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
            df.rename(columns=lambda x: x.lower(), inplace=True)
            
            # Ensure required indicators exist
            required_indicators = [
                'high', 'low', 'close', 'volume',
                f"ema_{self.config['entry_filters']['ema_period']}",
                'rsi_14', 'volume_ratio'
            ]
            
            for indicator in required_indicators:
                if indicator not in df.columns:
                    raise ValueError(f"Missing indicator {indicator} in {symbol} data")
                    
            return df
            
        except Exception as e:
            print(f"Error loading {symbol} data: {str(e)}")
            return None

    def validate_position_sizing(self, trade, vix_data):
        """Robust position sizing validation"""
        try:
            entry_date = trade['entry_date']
            if entry_date not in vix_data.index:
                return False, f"VIX data missing for {entry_date}"
                
            vix_value = vix_data.loc[entry_date, 'close']
            dr_cfg = self.config['dynamic_risk']
            
            # Determine expected risk percent
            if not dr_cfg['enabled'] or pd.isna(vix_value):
                expected_risk = self.config['risk_per_trade_percent']
            elif vix_value <= dr_cfg['vix_thresholds'][0]:
                expected_risk = dr_cfg['risk_percents'][0]
            elif vix_value <= dr_cfg['vix_thresholds'][1]:
                expected_risk = dr_cfg['risk_percents'][1]
            else:
                expected_risk = dr_cfg['risk_percents'][2]
                
            # Calculate actual risk taken
            risk_per_share = trade['entry_price'] - trade['initial_stop_loss']
            capital_at_risk = trade.get('capital_at_risk', risk_per_share * trade['shares'])
            
            if capital_at_risk <= 0:
                return False, "Zero/negative capital at risk"
                
            actual_risk_pct = (risk_per_share * trade['shares']) / capital_at_risk * 100
            
            if not np.isclose(actual_risk_pct, expected_risk, rtol=0.05):
                return False, f"Risk % mismatch: Expected {expected_risk}%, Actual {actual_risk_pct:.2f}% (VIX={vix_value})"
                
            return True, "Position sizing valid"
            
        except Exception as e:
            return False, f"Position sizing validation error: {str(e)}"

    def validate_all_trades(self):
        """Main validation function with comprehensive error handling"""
        try:
            trades_df = self.load_trade_data()
            vix_data = self.load_symbol_data(self.vix_symbol)
            
            if vix_data is None:
                raise ValueError("VIX data could not be loaded")
                
            results = []
            for _, trade in trades_df.iterrows():
                try:
                    symbol = trade['symbol']
                    symbol_data = self.load_symbol_data(symbol)
                    
                    if symbol_data is None:
                        results.append({
                            'trade_id': _,
                            'symbol': symbol,
                            'validation_status': 'Error',
                            'message': f"Could not load data for {symbol}"
                        })
                        continue
                        
                    validations = {
                        'setup': self.validate_setup_conditions(trade, symbol_data),
                        'entry': self.validate_entry(trade, symbol_data),
                        'exit': self.validate_exit(trade, symbol_data),
                        'position_sizing': self.validate_position_sizing(trade, vix_data),
                        'trailing_stop': self.validate_trailing_stop(trade, symbol_data)
                    }
                    
                    all_valid = all(v[0] for v in validations.values())
                    
                    results.append({
                        'trade_id': _,
                        'symbol': symbol,
                        'entry_date': trade['entry_date'],
                        'validation_status': 'Valid' if all_valid else 'Invalid',
                        **{f"{k}_validation": v[1] for k, v in validations.items()}
                    })
                    
                except Exception as e:
                    results.append({
                        'trade_id': _,
                        'symbol': trade.get('symbol', 'UNKNOWN'),
                        'validation_status': 'Error',
                        'message': f"Validation failed: {str(e)}"
                    })
                    continue
                    
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"Fatal error during validation: {str(e)}")
            return pd.DataFrame([{
                'validation_status': 'Error',
                'message': f"Validation could not complete: {str(e)}"
            }])

if __name__ == "__main__":
    config = {
        'data_folder': os.path.join('data', 'universal_processed', 'daily'),
        'vix_symbol': 'INDIAVIX',
        'slippage_on_entry_percent': 0.05,
        'slippage_on_exit_percent': 0.05,
        'entry_filters': {
            'use_volume_filter': True,
            'min_volume_ratio': 1.5,
            'use_ema_filter': True,
            'ema_period': 10,
            'use_rsi_filter': True,
            'rsi_ranges': [[0, 40], [70, 100]],
        },
        'trade_management': {
            'use_atr': True,
            'use_breakeven': True,
            'atr_period': 14,
            'atr_multiplier': 8.0,
            'breakeven_buffer_percent': 0.1,
            'dynamic_atr': {
                'enabled': True,
                'rsi_threshold': 60,
                'low_rsi_multiplier': 3.0,
                'high_rsi_multiplier': 8.0
            }
        },
        'dynamic_risk': {
            'enabled': True,
            'vix_thresholds': [15, 22],
            'risk_percents': [2.5, 0.5, 2.0]
        },
        'risk_per_trade_percent': 2.0
    }
    
    backtest_log_folder = "backtest_logs/candle_breakout_portfolio/latest_run"
    
    validator = BreakoutStrategyValidator(config, backtest_log_folder)
    results = validator.validate_all_trades()
    
    if not results.empty:
        print("\nValidation Summary:")
        print(results['validation_status'].value_counts())
        
        output_path = os.path.join(backtest_log_folder, "validation_report.csv")
        results.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to {output_path}")
    else:
        print("No validation results were generated")