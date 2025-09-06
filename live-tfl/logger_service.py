# ------------------------------------------------------------------------------------------------
# logger_service.py - The Centralized Scribe for the Trading Bot
# ------------------------------------------------------------------------------------------------
#
# This component handles all logging, both to the console and to detailed CSV files.
# By centralizing logging, we can easily control the verbosity and ensure that
# file writing does not block the main trading logic.
#
# NEW: Added support for 'in_trade_management_log' to provide detailed insights
#      into the dynamic exit strategy.
#
# ------------------------------------------------------------------------------------------------

import csv
import os
from datetime import datetime
import threading
import queue
import pytz

import config

class LoggerService:
    def __init__(self, verbose=False):
        self.verbose_logging = verbose
        self.log_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self.timezone = pytz.timezone(config.MARKET_TIMEZONE)
        
        self.csv_files = {}
        self.csv_writers = {}
        self.csv_headers = {
            'data_health_log': ['timestamp', 'event_type', 'symbol', 'status', 'details'],
            'scanner_log': ['timestamp', 'symbol', 'price', 'daily_rsi', 'is_rsi_ok', 'mvwap', 'is_mvwap_ok', 'pattern_found', 'is_setup_valid', 'rejection_reason'],
            'execution_log': ['timestamp', 'symbol', 'direction', 'status', 'details'],
            'trade_log': ['symbol', 'direction', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'quantity', 'pnl', 'exit_reason'],
            'in_trade_management_log': ['timestamp', 'symbol', 'update_type', 'old_sl', 'new_sl', 'current_price', 'current_r_value']
        }
        self.lock = threading.Lock()
        
        if self.verbose_logging:
            self._setup_csv_logging()

        self.worker_thread.start()

    def _setup_csv_logging(self):
        """Creates the log directory and initializes CSV files and writers."""
        log_dir = self.get_todays_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        
        for log_type, headers in self.csv_headers.items():
            filepath = os.path.join(log_dir, f"{log_type}.csv")
            file_exists = os.path.exists(filepath)
            
            # Use 'a+' to allow appending and reading, newline='' to handle line endings correctly
            f = open(filepath, 'a+', newline='', encoding='utf-8')
            self.csv_files[log_type] = f
            
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader() # Write header only if the file is new
            
            self.csv_writers[log_type] = writer

    def get_todays_log_dir(self):
        """Returns the path for today's log directory."""
        return os.path.join('logs', datetime.now(self.timezone).strftime('%Y-%m-%d'))

    def log_console(self, level, message):
        """Logs a message to the console with a timestamp and level."""
        timestamp = datetime.now(self.timezone).strftime('%H:%M:%S')
        print(f"{timestamp} | {level.upper():<7} | {message}")

    def log_to_csv(self, log_type, data_dict):
        """Puts a log message into the queue to be written to a CSV file."""
        if self.verbose_logging:
            self.log_queue.put({'type': log_type, 'data': data_dict})

    def _process_log_queue(self):
        """The target function for the worker thread to process log messages."""
        while True:
            try:
                log_item = self.log_queue.get()
                if log_item is None: # Shutdown signal
                    break
                
                log_type = log_item['type']
                data_dict = log_item['data']

                with self.lock:
                    if log_type in self.csv_writers:
                        try:
                            self.csv_writers[log_type].writerow(data_dict)
                            self.csv_files[log_type].flush() # Ensure it's written immediately
                        except Exception as e:
                            self.log_console("ERROR", f"Failed to write to {log_type}.csv: {e}")
                
                self.log_queue.task_done()
            except Exception as e:
                # This outer catch is for unexpected errors in the queue logic itself
                self.log_console("FATAL", f"Logger thread encountered an error: {e}")

    def shutdown(self):
        """Shuts down the logger service gracefully."""
        self.log_console("INFO", "Logger service shutting down...")
        self.log_queue.put(None) # Signal the worker thread to exit
        self.worker_thread.join(timeout=5) # Wait for the thread to finish
        
        with self.lock:
            for f in self.csv_files.values():
                f.close()
        print("Logger service shut down.")

