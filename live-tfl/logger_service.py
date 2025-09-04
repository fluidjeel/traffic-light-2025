# ------------------------------------------------------------------------------------------------
# logger_service.py - Centralized Logging for Console and CSV
# ------------------------------------------------------------------------------------------------
#
# This component acts as the central hub for all logging activities.
# - It receives log messages from all other components.
# - It prints formatted, high-level messages to the console for real-time monitoring.
# - When verbose mode is enabled, it writes detailed, structured logs to daily CSV files
#   for deep-dive analysis and debugging.
#
# ------------------------------------------------------------------------------------------------

import os
import csv
import queue
import threading
from datetime import datetime

class LoggerService:
    def __init__(self, verbose_logging=False):
        self.verbose_logging = verbose_logging
        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        self.log_dir = self.get_todays_log_dir()
        if self.verbose_logging:
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"13:54:03 | INFO   | Verbose logging is ON. Detailed logs will be saved to: {self.log_dir}")

        self.csv_files = {}
        self.csv_writers = {}
        self.csv_headers = {
            'data_health_log': ['timestamp', 'event_type', 'symbol', 'status', 'details'],
            'scanner_log': ['timestamp', 'symbol', 'price', 'daily_rsi', 'is_rsi_ok', 'mvwap', 'is_mvwap_ok', 'pattern_found', 'is_setup_valid', 'rejection_reason'],
            'execution_log': ['timestamp', 'symbol', 'direction', 'status', 'rejection_reason', 'qty_by_risk', 'qty_by_capital', 'final_qty', 'fill_price']
        }

        if self.verbose_logging:
            self._setup_csv_files()

        self.worker_thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self.worker_thread.start()

    def _setup_csv_files(self):
        """Initializes CSV files with headers."""
        for log_type, headers in self.csv_headers.items():
            file_path = os.path.join(self.log_dir, f"{log_type}.csv")
            # Open file and keep it open
            self.csv_files[log_type] = open(file_path, 'w', newline='', encoding='utf-8')
            writer = csv.DictWriter(self.csv_files[log_type], fieldnames=headers)
            writer.writeheader()
            self.csv_writers[log_type] = writer

    def _process_log_queue(self):
        """The target method for the worker thread to process logs."""
        while not self.stop_event.is_set() or not self.log_queue.empty():
            try:
                log_item = self.log_queue.get(timeout=1)
                if log_item is None: continue

                log_type = log_item['type']
                data = log_item['data']

                if log_type == 'console':
                    print(data)
                elif self.verbose_logging and log_type in self.csv_writers:
                    # --- FIX: Correctly call writerow on the DictWriter object ---
                    self.csv_writers[log_type].writerow(data)
                    self.csv_files[log_type].flush() # Ensure data is written immediately

            except queue.Empty:
                continue
            except Exception as e:
                print(f"CRITICAL: Logger thread encountered an error: {e}")

    def log_console(self, level, message):
        """Logs a message to the console."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"{timestamp} | {level.ljust(7)}| {message}"
        self.log_queue.put({'type': 'console', 'data': log_entry})

    def log_to_csv(self, log_type, data_dict):
        """Logs a dictionary of data to the specified CSV file."""
        if self.verbose_logging:
            # Add timestamp to every CSV log entry
            data_dict_with_ts = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            data_dict_with_ts.update(data_dict)
            self.log_queue.put({'type': log_type, 'data': data_dict_with_ts})
    
    def get_todays_log_dir(self):
        """Returns the path for today's log directory."""
        return os.path.join(os.getcwd(), 'logs', datetime.now().strftime('%Y-%m-%d'))

    def shutdown(self):
        """Gracefully shuts down the logging service."""
        self.log_queue.put(None) # Sentinel to unblock the queue
        self.stop_event.set()
        self.worker_thread.join(timeout=5)
        for f in self.csv_files.values():
            f.close()
        print(f"{datetime.now().strftime('%H:%M:%S')} | INFO   | Logger service shut down.")

