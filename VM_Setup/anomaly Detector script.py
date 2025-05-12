# -*- coding: utf-8 -*-

"""
Prometheus Time Series Anomaly Detection using LSTM, Isolation Forest, and Prophet.

This script fetches specified metrics from Prometheus, trains anomaly detection
models (LSTM Autoencoder, Isolation Forest, Prophet), identifies anomalies in
a recent test period, generates plots for analysis, pushes detection
metrics and anomaly markers to a Prometheus Pushgateway, and sends email
alerts based on configurable significance rules (consensus and persistence)
ONLY for anomalies detected within the last 30 minutes.
"""

# --- Core Libraries ---
import os
import logging
import warnings
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import socket # For network error handling

# --- Data Handling & ML ---
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
# Conditionally import TensorFlow to reduce noise if not used/installed later
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    # Configure logger early for potential import warnings
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow/Keras not found. LSTM analysis will be disabled.")
    Sequential, LSTM, Dense, Dropout = None, None, None, None # Define as None
    TENSORFLOW_AVAILABLE = False


# Defer Prophet import to handle potential absence
# from prophet import Prophet

# --- Plotting ---
import matplotlib.pyplot as plt

# --- Prometheus Integration ---
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# --- Environment Loading ---
from dotenv import load_dotenv
load_dotenv()

# --- Email Alerting ---
import smtplib
from email.message import EmailMessage

# --- Logging Configuration ---
# Ensure logger is configured if TF import didn't trigger it
if 'logger' not in locals():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


# --- Suppress Common Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='prophet') # Prophet UserWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # Numpy percentile warning

# --- Configuration ---
# || Prometheus Source ||
PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL', 'http://localhost:9090')
METRICS_TO_FETCH = {
    'count': os.environ.get('METRIC_COUNT', 'http_server_requests_seconds_count'),
    'sum': os.environ.get('METRIC_SUM', 'http_server_requests_seconds_sum'),
    'max': os.environ.get('METRIC_MAX', 'http_server_requests_seconds_max')
}
FETCH_HOURS = int(os.environ.get('FETCH_HOURS', 4))
QUERY_STEP = os.environ.get('QUERY_STEP', '60s')

# || Prometheus Pushgateway Target ||
PUSHGATEWAY_URL = os.environ.get('PUSHGATEWAY_URL', 'http://localhost:9091')
JOB_NAME = os.environ.get('JOB_NAME', 'anomaly_detector_job')

# || LSTM Model Settings ||
SEQ_LEN = int(os.environ.get('SEQ_LEN', 30))
TRAIN_SPLIT_RATIO = float(os.environ.get('TRAIN_SPLIT_RATIO', 0.8))
LSTM_EPOCHS = int(os.environ.get('LSTM_EPOCHS', 10))
LSTM_BATCH_SIZE = int(os.environ.get('LSTM_BATCH_SIZE', 32))
LSTM_VALIDATION_SPLIT = float(os.environ.get('LSTM_VALIDATION_SPLIT', 0.1))
ANOMALY_THRESHOLD_PERCENTILE = int(os.environ.get('ANOMALY_THRESHOLD_PERCENTILE', 95)) # For LSTM MAE

# || Isolation Forest Settings ||
IFOREST_CONTAMINATION_STR = os.environ.get('IFOREST_CONTAMINATION', 'auto')
try:
    IFOREST_CONTAMINATION = float(IFOREST_CONTAMINATION_STR)
    if not (0 < IFOREST_CONTAMINATION <= 0.5):
        logger.warning(f"IFOREST_CONTAMINATION value '{IFOREST_CONTAMINATION}' out of range (0, 0.5]. Using 'auto' (effective default 0.05).")
        IFOREST_CONTAMINATION = 'auto'
except ValueError:
    if IFOREST_CONTAMINATION_STR.lower() != 'auto':
        logger.warning(f"Invalid IFOREST_CONTAMINATION value '{IFOREST_CONTAMINATION_STR}'. Using 'auto' (effective default 0.05).")
    IFOREST_CONTAMINATION = 'auto'

# || Prophet Settings ||
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logger.info("Prophet library loaded successfully.")
except ImportError:
    logger.warning("Prophet library not found or failed to import. Prophet analysis will be disabled.")
    Prophet = None
    PROPHET_AVAILABLE = False

PROPHET_INTERVAL_WIDTH = float(os.environ.get('PROPHET_INTERVAL_WIDTH', 0.95))
PROPHET_TARGET_METRIC_KEY = 'count'
PROPHET_TARGET_METRIC = METRICS_TO_FETCH.get(PROPHET_TARGET_METRIC_KEY)
if PROPHET_AVAILABLE and not PROPHET_TARGET_METRIC:
     logger.error(f"Prophet target metric key '{PROPHET_TARGET_METRIC_KEY}' not found in METRICS_TO_FETCH. Prophet disabled.")
     PROPHET_AVAILABLE = False # Disable Prophet if target is missing

# || General Settings ||
RANDOM_STATE = 42
PLOT_OUTPUT_DIR = os.environ.get('PLOT_OUTPUT_DIR', 'anomaly_plots')

# || Email Alerting Configuration ||
# --- Use provided credentials via environment variables ---
SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.mailersend.net')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587)) # 587 is standard for TLS
SMTP_USER = os.environ.get('SMTP_USER', 'MS_eWsEnS@test-z0vklo667qpl7qrx.mlsender.net')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', 'YOUR_PASSWORD') # Use secure storage for sensitive data
EMAIL_SENDER = os.environ.get('EMAIL_SENDER', SMTP_USER) # Usually same as user for MailerSend

# Hardcoding recipient as previously requested.
# Recommended: Use os.environ.get('EMAIL_RECIPIENTS') instead for flexibility.
EMAIL_RECIPIENTS = "mahdi.bouafif@gmail.com"
# logger.warning("Email recipient is hardcoded to 'mahdi.bouafif@gmail.com'. Consider using environment variable EMAIL_RECIPIENTS.")

# || Alerting Rules ||
# Rule 1: Consensus
MIN_MODELS_FOR_ALERT = int(os.environ.get('MIN_MODELS_FOR_ALERT', 2)) # N models agree
# Rule 2: Persistence
ENABLE_PERSISTENCE_ALERT = os.environ.get('ENABLE_PERSISTENCE_ALERT', 'true').lower() == 'true'
MIN_CONSECUTIVE_ANOMALIES = int(os.environ.get('MIN_CONSECUTIVE_ANOMALIES', 3)) # M consecutive anomalies (any model)
PERSISTENCE_WINDOW_SIZE = int(os.environ.get('PERSISTENCE_WINDOW_SIZE', 5)) # Look at last Y steps
MIN_ANOMALIES_IN_WINDOW = int(os.environ.get('MIN_ANOMALIES_IN_WINDOW', 3)) # X anomalies in last Y steps
# Rule 3: Recency Filter
ALERT_RECENCY_MINUTES = int(os.environ.get('ALERT_RECENCY_MINUTES', 30)) # Only alert if anomaly within this many minutes

# --- Prometheus Client Initialization ---
registry = CollectorRegistry()
combined_anomaly_marker = Gauge(
    'anomaly_detected_marker',
    'Marker set to 1 for anomalies in the test period, labeled by models, timestamp, and job',
    ['models', 'timestamp_unix', 'job'],
    registry=registry
)
# Only define LSTM gauges if TF is available
if TENSORFLOW_AVAILABLE:
    lstm_mae_gauge = Gauge(
        'lstm_reconstruction_mae_latest',
        'LSTM Reconstruction Mean Absolute Error (latest point in test set)',
        ['job'],
        registry=registry
    )
    lstm_threshold_gauge = Gauge(
        'lstm_mae_threshold',
        'LSTM MAE Threshold used for anomaly detection (based on test set errors)',
        ['job', 'model'],
        registry=registry
    )
    lstm_anomaly_latest_marker = Gauge(
        'lstm_anomaly_detected_latest',
        'Marker indicating if LSTM detected an anomaly at the latest time point in the test set (1=yes, 0=no)',
        ['job', 'model'],
        registry=registry
    )


# --- Helper Functions ---

def fetch_prometheus_metric(metric_name: str, start_time: datetime, end_time: datetime, step: str = '60s') -> list:
    """Fetches metric data from Prometheus using the query_range API."""
    api_endpoint = '/api/v1/query_range'
    params = {
        'query': metric_name,
        'start': int(start_time.timestamp()),
        'end': int(end_time.timestamp()),
        'step': step
    }
    url = f'{PROMETHEUS_URL}{api_endpoint}'
    logger.info(f"Fetching range query '{metric_name}' from {start_time} to {end_time} with step {step}...")

    try:
        # Increased timeout for potentially larger queries
        res = requests.get(url, params=params, timeout=180)
        res.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = res.json()

        if data.get('status') == 'success':
            result_data = data.get('data', {}).get('result', [])
            if result_data:
                # Log details about the first series if available
                if len(result_data[0].get('values',[])) > 0:
                    num_points = len(result_data[0]['values'])
                    logger.info(f"Successfully fetched {len(result_data)} series for '{metric_name}'. First series has {num_points} points.")
                else:
                     logger.info(f"Successfully fetched {len(result_data)} series for '{metric_name}', but the first series has no values.")
                return result_data
            else:
                logger.warning(f"No data returned for query '{metric_name}' in the specified range [{start_time} - {end_time}].")
                return []
        else:
            error_type = data.get('errorType', 'N/A')
            error_msg = data.get('error', 'N/A')
            logger.warning(f"Prometheus query unsuccessful for '{metric_name}'. Status: {data.get('status', 'N/A')}, Type: {error_type}, Error: {error_msg}")
            return []
    except requests.exceptions.Timeout:
        timeout_value = 180 # Hardcoding timeout value used in request for logging
        logger.error(f"Timeout error ({timeout_value}s) fetching Prometheus data for '{metric_name}' from {url}.")
        return []
    except requests.exceptions.ConnectionError as e:
         logger.error(f"Connection error fetching Prometheus data for '{metric_name}' from {url}: {e}")
         return []
    except requests.exceptions.RequestException as e:
        # Includes HTTPError, etc.
        logger.error(f"Request error fetching Prometheus data for '{metric_name}': {e}")
        # Log response body if available and potentially useful (be careful with large responses)
        try:
            if e.response is not None:
                 logger.error(f"Response status: {e.response.status_code}, Body: {e.response.text[:500]}...") # Log first 500 chars
        except Exception:
            pass # Ignore errors during error logging enhancement
        return []
    except Exception as e:
        logger.error(f"Unexpected error occurred while fetching Prometheus data for '{metric_name}': {e}", exc_info=True)
        return []


def parse_range_data(result_list: list, metric_name_label: str) -> pd.DataFrame:
    """Parses Prometheus range query result list into a Pandas DataFrame (UTC index)."""
    all_series_df = []
    if not result_list:
        # This is expected if fetch returned empty, just log debug
        logger.debug(f"Received empty result list for parsing '{metric_name_label}'")
        return pd.DataFrame(columns=[metric_name_label], index=pd.DatetimeIndex([], tz='UTC'))

    num_parsed_series = 0
    for series in result_list:
        metric_labels = series.get('metric', {})
        values = series.get('values')

        if values: # Check if values list is not empty
            try:
                df = pd.DataFrame(values, columns=['unix_ts', metric_name_label])
                # Convert timestamp to UTC datetime
                df['timestamp'] = pd.to_datetime(df['unix_ts'], unit='s', utc=True)
                # Convert value to numeric, coercing errors to NaN
                df[metric_name_label] = pd.to_numeric(df[metric_name_label], errors='coerce')
                # Set timestamp as index, drop original unix_ts, remove rows with NaN values (from coercion)
                df = df.drop(columns=['unix_ts']).set_index('timestamp').dropna(subset=[metric_name_label]) # Drop based on target column

                if not df.empty:
                    # Add labels as multi-index columns if needed later, or just keep the value
                    # For now, we assume we want to average series with the same name
                    all_series_df.append(df)
                    num_parsed_series += 1
                else:
                    logger.debug(f"Series for '{metric_name_label}' with labels {metric_labels} became empty after NA drop.")

            except Exception as e:
                logger.error(f"Error parsing series for '{metric_name_label}' with labels {metric_labels}: {e}", exc_info=True)
        else:
             # Log if a series is present but has no values
             logger.debug(f"No values found in series for '{metric_name_label}' with labels {metric_labels}")

    if not all_series_df:
        logger.warning(f"No valid data series parsed from result list for '{metric_name_label}' (parsed {num_parsed_series} series initially).")
        return pd.DataFrame(columns=[metric_name_label], index=pd.DatetimeIndex([], tz='UTC'))

    try:
        # Concatenate potentially multiple series (e.g., from different pods)
        # `join='outer'` keeps all timestamps, `axis=1` places series side-by-side
        combined_df = pd.concat(all_series_df, axis=1, join='outer')

        # Handle multiple series for the *same* metric name (e.g., http_requests_total from 3 pods)
        # Check if the specific metric name appears multiple times as a column name
        if isinstance(combined_df.columns, pd.Index): # Ensure it's an index object
             metric_cols = combined_df.columns[combined_df.columns == metric_name_label]
             if len(metric_cols) > 1:
                 logger.warning(f"Multiple series found for target metric '{metric_name_label}'. Averaging values across these series.")
                 # Select only the columns for the target metric, calculate row-wise mean, keep as DataFrame
                 combined_df[metric_name_label] = combined_df[metric_cols].mean(axis=1)
                 # Drop the original duplicated columns to avoid confusion
                 combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='last')]


        # Ensure the target column exists after potential averaging
        if metric_name_label not in combined_df.columns:
             logger.error(f"Target metric '{metric_name_label}' not found in columns after processing. Columns: {combined_df.columns}")
             return pd.DataFrame(columns=[metric_name_label], index=pd.DatetimeIndex([], tz='UTC'))

        # Select only the target metric column, ensure float type, sort by time
        final_df = combined_df[[metric_name_label]].astype(float).sort_index()

        # Check for duplicate timestamps (can happen with 'outer' join or Prometheus step issues)
        duplicates = final_df.index.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate timestamps for '{metric_name_label}' after concat/avg. Averaging values for duplicates.")
            # Group by index and take the mean for duplicate timestamps
            final_df = final_df.groupby(final_df.index).mean()

        # Final check for empty dataframe
        if final_df.empty:
            logger.warning(f"DataFrame for '{metric_name_label}' is empty after all parsing steps.")
            return final_df # Return empty frame

        logger.info(f"Successfully parsed '{metric_name_label}'. Final shape: {final_df.shape}. Index range (UTC): {final_df.index.min()} to {final_df.index.max()}")
        return final_df

    except Exception as e:
        logger.error(f"Error during final processing/concatenation for '{metric_name_label}': {e}", exc_info=True)
        # Return an empty DataFrame on error
        return pd.DataFrame(columns=[metric_name_label], index=pd.DatetimeIndex([], tz='UTC'))


def create_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Creates sequences of data for time series forecasting models like LSTMs."""
    X, y = [], []
    if not isinstance(seq_len, int) or seq_len <= 0:
        logger.error(f"Sequence length must be a positive integer. Got: {seq_len}")
        num_features = data.shape[1] if hasattr(data, 'shape') and data.ndim == 2 else 1
        return np.array([]).reshape(0, seq_len, num_features), np.array([])

    if data is None or data.ndim != 2: # Expect 2D array (samples, features)
        logger.error(f"Invalid input data for sequence creation. Expected 2D numpy array, got: {type(data)} with shape {getattr(data, 'shape', 'N/A')}")
        num_features = data.shape[1] if hasattr(data, 'shape') and data.ndim == 2 else 1
        return np.array([]).reshape(0, seq_len, num_features), np.array([])

    if len(data) <= seq_len:
        logger.warning(f"Not enough data ({len(data)}) for sequence length ({seq_len}). Need > {seq_len} points.")
        num_features = data.shape[1]
        return np.array([]).reshape(0, seq_len, num_features), np.array([])

    num_sequences = len(data) - seq_len
    num_features = data.shape[1]
    logger.debug(f"Creating {num_sequences} sequences of length {seq_len} from data shape {data.shape}.")

    # Pre-allocate numpy arrays for potentially better performance
    X_np = np.zeros((num_sequences, seq_len, num_features), dtype=data.dtype)
    y_np = np.zeros((num_sequences, num_features), dtype=data.dtype)

    for i in range(num_sequences):
        X_np[i] = data[i:(i + seq_len)]
        y_np[i] = data[i + seq_len]

    if X_np.size == 0 or y_np.size == 0:
        logger.warning("Sequence creation resulted in empty NumPy arrays.")
        return np.array([]).reshape(0, seq_len, num_features), np.array([])

    logger.debug(f"Created sequences: X shape {X_np.shape}, y shape {y_np.shape}")
    return X_np, y_np


def save_plot(fig, filename: str):
    """Saves a matplotlib figure to the configured plot directory and closes it."""
    try:
        # Ensure directory exists
        os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(PLOT_OUTPUT_DIR, filename)
        logger.info(f"Saving plot to {filepath}")
        # Use tight layout and specify resolution
        fig.savefig(filepath, bbox_inches='tight', dpi=100)
    except Exception as e:
        logger.error(f"Failed to save plot {filepath}: {e}", exc_info=True)
    finally:
        # Ensure figure is closed to free memory, even if saving failed
        plt.close(fig)

def set_combined_anomaly_metrics(gauge: Gauge, combined_anomalies: dict, job_name: str) -> int:
    """Sets the combined anomaly marker gauge for anomalies detected in the current cycle."""
    if not combined_anomalies:
        logger.info("No combined anomalies identified in this cycle to set metrics for.")
        return 0

    set_count = 0
    logger.info(f"Setting {len(combined_anomalies)} unique anomaly timestamps on the gauge for job '{job_name}'...")

    # Sort timestamps for potentially more ordered metric exposure (though Prometheus doesn't guarantee order)
    sorted_timestamps = sorted(combined_anomalies.keys())

    for ts in sorted_timestamps:
        model_list = combined_anomalies[ts]
        try:
            # Ensure models are unique and sorted for consistent labeling
            model_str = ",".join(sorted(list(set(model_list))))
            # Convert timestamp to Unix integer string
            ts_unix_str = str(int(ts.timestamp())) # Assumes ts is timezone-aware UTC
            # Set the gauge value to 1 for this specific combination of labels
            gauge.labels(models=model_str, timestamp_unix=ts_unix_str, job=job_name).set(1)
            logger.debug(f"Set anomaly marker=1 | job='{job_name}', models='{model_str}', timestamp={ts} ({ts_unix_str})")
            set_count += 1
        except Exception as e:
            # Log details if setting a specific metric fails
            model_str_err = ",".join(model_list) # Use original list for error logging
            ts_err_str = str(ts)
            logger.error(f"Failed to set anomaly marker value | job='{job_name}', models='{model_str_err}', ts={ts_err_str}: {e}", exc_info=True)

    logger.info(f"Finished setting combined anomaly markers for this cycle. Markers set: {set_count}")
    return set_count


# --- Email Alerting Logic ---
import pandas as pd
import logging
from datetime import timezone
from collections import defaultdict # Useful for aggregation

logger = logging.getLogger(__name__) # Replace with your actual logger setup

def identify_significant_anomalies_combined(
    combined_anomalies: dict[pd.Timestamp, list[str]],
    test_timestamps: pd.DatetimeIndex,
    min_models: int,
    enable_persistence: bool,
    min_consecutive: int,  # Now refers to consecutive *minutes* with anomalies
    window_size: int,      # Now refers to a window of *minutes*
    min_in_window: int     # Now refers to min *minutes* with anomalies in the window
) -> dict[pd.Timestamp, list[str]]:
    """
    Filters combined anomalies based on minute-level persistence AND consensus rules.

    An anomaly timestamp is considered significant ONLY IF:
    1. It belongs to a minute that is part of a sequence meeting a minute-level
       persistence rule (consecutive minutes or window density of minutes).
    2. The *last precise anomaly timestamp* within that triggering minute ALSO
       meets the consensus criterion (>= min_models).

    Returns a dictionary containing only the significant anomalies {Timestamp: [models]}.
    Keys are the precise timestamps identified.
    """
    significant_alerts = {}
    if not combined_anomalies or not enable_persistence:
        if not enable_persistence:
            logger.info("Persistence checks are disabled. No anomalies will be marked significant.")
        else:
            logger.info("No combined anomalies provided.")
        return significant_alerts
    if test_timestamps.empty:
        logger.warning("Persistence checks require non-empty test_timestamps.")
        return significant_alerts # Cannot check persistence

    # --- Timezone Standardization (same as before) ---
    try:
        combined_anomalies_utc = {ts.tz_convert('UTC') if ts.tz is not None else ts.tz_localize('UTC'): models
                                  for ts, models in combined_anomalies.items()}
        if test_timestamps.tz is None:
            logger.warning("Test timestamps were timezone-naive. Assuming UTC for persistence checks.")
            test_timestamps = test_timestamps.tz_localize('UTC')
        elif test_timestamps.tz != timezone.utc:
            logger.warning(f"Test timestamps had timezone {test_timestamps.tz}. Converting to UTC.")
            test_timestamps = test_timestamps.tz_convert('UTC')
    except Exception as tz_err:
         logger.error(f"Error standardizing timezones: {tz_err}. Cannot proceed.", exc_info=True)
         return significant_alerts

    # --- Pre-process: Aggregate Anomalies per Minute ---
    logger.info("Aggregating anomalies per minute...")
    # minute_data structure: {minute_start_ts: (has_anomaly: bool, last_precise_ts: Timestamp | None, models_at_last_ts: list | None)}
    minute_data = defaultdict(lambda: (False, None, None))
    temp_minute_agg = defaultdict(list) # {minute_start: [(ts1, models1), (ts2, models2)...]}

    # Group anomalies by minute
    for ts, models in combined_anomalies_utc.items():
        minute_start = ts.floor('min')
        temp_minute_agg[minute_start].append((ts, models))

    # Determine last anomaly and models for each minute that had anomalies
    for minute_start, ts_models_list in temp_minute_agg.items():
        ts_models_list.sort(key=lambda x: x[0]) # Sort by timestamp within the minute
        last_ts, last_models = ts_models_list[-1]
        minute_data[minute_start] = (True, last_ts, last_models)

    # --- Generate Full Minute Index for the Period ---
    if test_timestamps.empty: # Should have been caught earlier, but double check
        logger.warning("Cannot generate minute index from empty test_timestamps.")
        return significant_alerts

    # Ensure the minute range covers all potential test timestamps
    analysis_start_minute = test_timestamps.min().floor('min')
    analysis_end_minute = test_timestamps.max().floor('min')
    # Include the end minute itself
    minute_index = pd.date_range(start=analysis_start_minute, end=analysis_end_minute, freq='min', tz='UTC')

    logger.info(f"Generated minute index from {analysis_start_minute} to {analysis_end_minute} ({len(minute_index)} minutes).")


    # --- Minute-Level Persistence Checks ---
    required_minutes = max(min_consecutive, window_size)
    if len(minute_index) < required_minutes:
         logger.warning(f"Persistence checks skipped: Not enough minutes ({len(minute_index)}) in the analysis range for rules (min_consecutive={min_consecutive}, window_size={window_size}).")
         return significant_alerts

    logger.info(f"Checking minute-level persistence (Consecutive >= {min_consecutive} mins OR Window >= {min_in_window}/{window_size} mins) AND Consensus (>= {min_models} models)...")

    consecutive_minute_count = 0
    # Stores *indices* relative to minute_index for anomalous minutes in the window
    window_minute_indices = []

    for i, current_minute_start in enumerate(minute_index):
        has_anomaly_this_minute, last_precise_ts, models_at_last_ts = minute_data[current_minute_start]

        consensus_met_at_last_ts = False
        unique_models = set()
        if has_anomaly_this_minute and models_at_last_ts:
            unique_models = set(models_at_last_ts)
            consensus_met_at_last_ts = len(unique_models) >= min_models

        # --- Check Consecutive Minute Rule ---
        if has_anomaly_this_minute:
            consecutive_minute_count += 1
            if consecutive_minute_count >= min_consecutive:
                # Check consensus for the representative timestamp of *this* minute
                if consensus_met_at_last_ts:
                    if last_precise_ts not in significant_alerts: # Use precise timestamp as key
                        significant_alerts[last_precise_ts] = sorted(list(unique_models))
                        logger.debug(f"Significant anomaly by Minute Persistence (Consecutive) AND Consensus: {last_precise_ts} (from minute {current_minute_start}, models {significant_alerts[last_precise_ts]})")
                # else: logger.debug(f"Consecutive minute rule met at {current_minute_start}, but consensus not met at its last anomaly {last_precise_ts} ({len(unique_models)}<{min_models})")
        else:
            consecutive_minute_count = 0 # Reset counter

        # --- Check Window Minute Rule ---
        # Remove indices from the window that are no longer within the lookback period (in minutes)
        window_start_index = i - window_size + 1
        window_minute_indices = [idx for idx in window_minute_indices if idx >= window_start_index]

        # Add current *minute index* if anomalous
        if has_anomaly_this_minute:
            window_minute_indices.append(i)

        # Check if the number of *anomalous minutes* in the window meets the threshold
        if len(window_minute_indices) >= min_in_window:
             # Rule met for the window ending at this minute.
             # Check consensus for the representative timestamp of *this* minute, only if this minute *itself* had an anomaly.
             if has_anomaly_this_minute and consensus_met_at_last_ts:
                 if last_precise_ts not in significant_alerts: # Use precise timestamp as key
                      significant_alerts[last_precise_ts] = sorted(list(unique_models))
                      logger.debug(f"Significant anomaly by Minute Persistence (Window Density) AND Consensus: {last_precise_ts} (from minute {current_minute_start}, models {significant_alerts[last_precise_ts]})")
             #elif has_anomaly_this_minute: logger.debug(f"Window density rule met ending {current_minute_start}, but consensus not met at its last anomaly {last_precise_ts} ({len(unique_models)}<{min_models})")

    num_significant = len(significant_alerts)
    logger.info(f"Found {num_significant} significant anomaly timestamps meeting minute-level persistence AND consensus criteria.")
    # Return dictionary with precise UTC timestamps as keys
    return significant_alerts

def send_anomaly_email(significant_anomalies: dict[pd.Timestamp, list[str]], job_name_email: str):
    """
    Formats and sends an email alert if significant anomalies are found.
    Expects significant_anomalies keys to be UTC Timestamps.
    Includes a reformulated suggestion for rollback.
    """
    if not significant_anomalies:
        logger.info("No recent significant anomalies identified meeting criteria. No email will be sent.")
        return

    # --- Validate Email Configuration ---
    if not EMAIL_RECIPIENTS: # Check if recipient list is non-empty
        logger.error("EMAIL_RECIPIENTS is blank or missing. Cannot send email.")
        return
    # Simple parsing of comma-separated list
    recipient_list = [email.strip() for email in EMAIL_RECIPIENTS.split(',') if email.strip()]
    if not recipient_list:
        logger.error("No valid email recipients found after parsing EMAIL_RECIPIENTS.")
        return
    # Check essential SMTP creds
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD]):
        logger.error("Incomplete SMTP configuration (HOST, PORT, USER, PASSWORD missing). Cannot send email.")
        return

    # --- Format the Email Content ---
    alert_rule_desc = f"Consensus (>= {MIN_MODELS_FOR_ALERT} models)"
    if ENABLE_PERSISTENCE_ALERT:
        alert_rule_desc += f" OR Persistence (>= {MIN_CONSECUTIVE_ANOMALIES} consecutive OR >= {MIN_ANOMALIES_IN_WINDOW}/{PERSISTENCE_WINDOW_SIZE} in window)"
    # Add recency to the description
    alert_rule_desc += f", occurring within the last {ALERT_RECENCY_MINUTES} minutes."

    subject = f"Anomaly Alert [{job_name_email}]: Recent Significant Deviations Detected"
    body_lines = [
        f"Recent significant anomalies detected by the '{job_name_email}' monitoring job.",
        f"Alert Rule Triggered: {alert_rule_desc}",
        f"Detection Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Total Recent Significant Timestamps: {len(significant_anomalies)}",
        "\n--- Recent Significant Anomaly Details ---",
    ]

    # Sort anomalies by timestamp for readability
    sorted_timestamps = sorted(significant_anomalies.keys())

    for ts in sorted_timestamps:
        models_str = ", ".join(significant_anomalies[ts])
        # Format timestamp clearly (already UTC)
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S %Z')
        body_lines.append(f"- Timestamp: {ts_str} | Models Flagging: {models_str}")

    body_lines.append("\nPlease investigate the system behavior around these times.")

    # *** ADDED REFORMULATED SUGGESTION HERE ***
    reformulated_suggestion = "As a potential mitigation step, you may want to consider rolling back to a previous version using the RollBack-Tool available at: http://98.66.179.208:5000/"
    body_lines.append(f"\n{reformulated_suggestion}")
    # *** END OF ADDED SUGGESTION ***

    body_lines.append(f"\nPlots may be available in: {PLOT_OUTPUT_DIR}") # Reference plot location

    body = "\n".join(body_lines)

    # --- Construct and Send Email ---
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = ", ".join(recipient_list) # Join the list for the 'To' header
    msg.set_content(body)

    logger.info(f"Attempting to send anomaly alert email via {SMTP_HOST}:{SMTP_PORT} to: {', '.join(recipient_list)}")
    try:
        # Use SMTP_SSL for port 465, STARTTLS for others (e.g., 587, 2525)
        if SMTP_PORT == 465:
             with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server: # Add timeout
                # server.set_debuglevel(1) # Uncomment for verbose SMTP debugging
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        else: # Assume STARTTLS
            # Add timeout to SMTP connection
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
                # server.set_debuglevel(1) # Uncomment for verbose SMTP debugging
                server.ehlo() # Identify client
                server.starttls() # Upgrade to secure connection
                server.ehlo() # Re-identify over secure connection
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        logger.info("Anomaly alert email sent successfully.")

    except smtplib.SMTPAuthenticationError:
        logger.error(f"SMTP Authentication failed for user {SMTP_USER} on {SMTP_HOST}. Check credentials/settings (e.g., App Password).")
    except smtplib.SMTPConnectError:
        logger.error(f"Failed to connect to SMTP server {SMTP_HOST}:{SMTP_PORT}. Check host/port/firewall.")
    except smtplib.SMTPSenderRefused:
         logger.error(f"Sender address {EMAIL_SENDER} refused by the server {SMTP_HOST}.")
    except smtplib.SMTPRecipientsRefused as e:
         # Log which recipients were refused if possible
         logger.error(f"One or more recipient addresses refused by {SMTP_HOST}: {e.recipients}")
    except (socket.gaierror, socket.timeout) as net_err:
         # Catch network resolution and timeout errors
         logger.error(f"Network error communicating with SMTP host {SMTP_HOST}: {net_err}")
    except Exception as e:
        # Catch any other unexpected errors during email sending
        logger.error(f"An unexpected error occurred while sending email via {SMTP_HOST}: {e}", exc_info=True)


# --- Main Detection Logic ---

def run_anomaly_detection(job_label: str):
    """Executes one full cycle of anomaly detection: fetch, preprocess, model, plot, push, alert."""
    logger.info(f"--- Starting Anomaly Detection Cycle for Job: {job_label} ---")
    start_cycle_time = datetime.now(timezone.utc)

    # 1. Data Fetching
    logger.info("Step 1: Fetching data from Prometheus...")
    end_fetch_time = start_cycle_time
    # Fetch slightly more to ensure sequence creation works even with sparse data near the start
    # Also consider QUERY_STEP when calculating buffer
    try:
         fetch_step_seconds = pd.to_timedelta(QUERY_STEP).total_seconds()
    except ValueError:
         logger.warning(f"Invalid QUERY_STEP '{QUERY_STEP}'. Using default 60s for buffer calculation.")
         fetch_step_seconds = 60
    buffer_minutes = max(SEQ_LEN * fetch_step_seconds / 60, 15) # Ensure at least 15 min buffer
    start_fetch_time = end_fetch_time - timedelta(hours=FETCH_HOURS) - timedelta(minutes=buffer_minutes)

    metric_dataframes = {}
    fetch_successful = True # Assume success unless a critical error occurs
    for key, metric_name in METRICS_TO_FETCH.items():
        raw_data = fetch_prometheus_metric(metric_name, start_fetch_time, end_fetch_time, step=QUERY_STEP)
        if raw_data is None: # Indicates a critical fetch error, stop processing this cycle
             # fetch_prometheus_metric logs the critical error
             fetch_successful = False
             break # Stop fetching other metrics for this cycle
        metric_df = parse_range_data(raw_data, metric_name)
        # Allow empty DFs for non-critical metrics, but log a warning
        if metric_df.empty:
            logger.warning(f"Parsing resulted in empty DataFrame for {key} ('{metric_name}'). This might affect models relying on it.")
        metric_dataframes[key] = metric_df

    if not fetch_successful:
         logger.critical("Aborting cycle due to critical data fetching errors.")
         return # Exit the function early

    # 2. Data Preparation & Preprocessing
    logger.info("Step 2: Preparing and preprocessing data...")
    valid_dfs = [df for key, df in metric_dataframes.items() if not df.empty]
    if not valid_dfs:
        logger.error("No valid data obtained from Prometheus for ANY specified metric after parsing. Cannot proceed.")
        return

    try:
        # Use 'outer' join to keep all time points, then handle NaNs
        combined_df = pd.concat(valid_dfs, axis=1, join='outer')
        # Forward fill first to propagate last known values, then backfill for initial NaNs
        combined_df = combined_df.ffill().bfill()
        # Drop any rows where *all* values might still be NaN (if all inputs were empty at that time)
        combined_df = combined_df.dropna(how='all')
         # Optionally drop rows where *any* value is NaN if models require complete data
        combined_df = combined_df.dropna(how='any')

    except Exception as e:
        logger.error(f"Error during DataFrame concatenation or filling: {e}", exc_info=True)
        return


    # Check for sufficient data *after* cleaning
    min_required_points = SEQ_LEN + 5 # Need sequence length + buffer for train/test split
    if combined_df.empty or len(combined_df) < min_required_points:
        logger.error(f"Insufficient data after combining and cleaning ({len(combined_df)} points). Requires at least {min_required_points}. Aborting cycle.")
        return
    logger.info(f"Combined data ready. Shape: {combined_df.shape}. Time range (UTC): {combined_df.index.min()} to {combined_df.index.max()}")

    # Scaling
    numeric_cols = combined_df.select_dtypes(include=np.number).columns
    scaled_data = None
    scaler = None
    if not numeric_cols.empty:
        logger.info(f"Scaling numeric columns: {list(numeric_cols)}")
        scaler = MinMaxScaler()
        # Ensure we only scale numeric columns
        try:
            scaled_data = scaler.fit_transform(combined_df[numeric_cols])
            # Check if scaling produced NaNs (e.g., column had zero variance)
            if np.isnan(scaled_data).any():
                logger.error("NaN values found after scaling. Check input data variance. Aborting.")
                # Consider attempting to impute NaNs instead of aborting?
                # For now, aborting is safer.
                return
        except Exception as scale_err:
             logger.error(f"Error during data scaling: {scale_err}", exc_info=True)
             return
    else:
        logger.error("No numeric columns found in combined data for scaling. Aborting.")
        return

    # Create Sequences
    X, y = create_sequences(scaled_data, SEQ_LEN)
    # create_sequences now returns shaped empty arrays on error/insufficient data
    if X.size == 0 or y.size == 0:
        logger.error("Failed to create valid sequences from the data (returned empty arrays). Aborting.")
        return

    # 3. Train/Test Split
    logger.info("Step 3: Splitting data into Training and Test sets...")
    X_train, y_train, X_test, y_test = None, None, None, None
    test_timestamps = pd.DatetimeIndex([], tz='UTC') # Initialize as empty UTC DatetimeIndex

    # Calculate split index based on the number of sequences
    train_size = int(TRAIN_SPLIT_RATIO * len(X))
    # Ensure there's at least one sample in test set if possible, and train set isn't empty
    min_test_samples = 1
    if train_size >= len(X) - min_test_samples:
        train_size = max(0, len(X) - min_test_samples) # Ensure test set has at least min_test_samples

    if train_size == 0 and len(X) > min_test_samples: # If split ratio is too small
        train_size = 1 # Need at least one sample for training
        logger.warning(f"Train split resulted in 0 samples, adjusting train_size to 1.")


    if 0 < train_size < len(X): # Check if split is valid
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        logger.info(f"Split complete. Train sequences: {X_train.shape[0]}, Test sequences: {X_test.shape[0]}")

        # Map sequence indices back to original DataFrame timestamps for the *test* set
        test_start_index_in_df = train_size + SEQ_LEN # Index in original df corresponding to first y_test value
        test_end_index_in_df = test_start_index_in_df + len(y_test) # Exclusive end index

        # Ensure indices are within the bounds of the original combined_df index
        if test_start_index_in_df < len(combined_df.index) and test_end_index_in_df <= len(combined_df.index):
            test_timestamps = combined_df.index[test_start_index_in_df:test_end_index_in_df] # Slicing preserves UTC timezone

            # Sanity check lengths (should match y_test length)
            if len(test_timestamps) != len(y_test):
                logger.warning(f"Timestamp mapping length mismatch! Test Timestamps: {len(test_timestamps)}, y_test: {len(y_test)}. Reconciling to minimum length.")
                min_len = min(len(test_timestamps), len(y_test))
                test_timestamps = test_timestamps[:min_len]
                y_test = y_test[:min_len]
                X_test = X_test[:min_len] # Ensure X_test matches too
                logger.info(f"Adjusted test set lengths to {min_len}.")

            if not test_timestamps.empty:
                 # Ensure test_timestamps are UTC after slicing/reconciliation
                 if test_timestamps.tz is None:
                      logger.warning("Test timestamps became timezone-naive after slicing. Assuming UTC.")
                      test_timestamps = test_timestamps.tz_localize('UTC')
                 elif test_timestamps.tz != timezone.utc:
                      test_timestamps = test_timestamps.tz_convert('UTC')
                 logger.info(f"Test Period Timestamps (UTC): {test_timestamps.min()} to {test_timestamps.max()}")
            else:
                # This can happen if reconciliation leads to zero length
                logger.error("Test timestamps became empty after reconciliation. Cannot proceed with testing.")
                X_test, y_test = None, None
        else:
             logger.error(f"Calculated test indices [{test_start_index_in_df}:{test_end_index_in_df}] are out of bounds for DataFrame index length {len(combined_df.index)}. Cannot map test timestamps.")
             X_test, y_test = None, None
             test_timestamps = pd.DatetimeIndex([], tz='UTC') # Ensure it's empty

    else:
         # This case handles invalid TRAIN_SPLIT_RATIO or insufficient total sequences
         logger.error(f"Cannot create a valid train/test split. Train size: {train_size}, Total sequences: {len(X)}. Check TRAIN_SPLIT_RATIO and data length. Aborting modeling.")
         X_train, y_train, X_test, y_test = None, None, None, None
         test_timestamps = pd.DatetimeIndex([], tz='UTC') # Ensure it's empty


    # 4. Model Training & Anomaly Detection
    # Initialize anomaly timestamp collections (ensure UTC)
    lstm_anomalies_ts = pd.DatetimeIndex([], tz='UTC')
    iforest_anomalies_ts = pd.DatetimeIndex([], tz='UTC')
    prophet_anomalies_ts = pd.DatetimeIndex([], tz='UTC')

    # --- LSTM Autoencoder ---
    if TENSORFLOW_AVAILABLE and X_train is not None and y_train is not None and X_test is not None and y_test is not None and not test_timestamps.empty:
        logger.info("--- Running LSTM Autoencoder ---")
        model_lstm = None
        errors_lstm = np.array([])
        threshold_lstm = np.nan # Use NaN for unset threshold
        anomalies_lstm_bool = np.array([], dtype=bool)
        try:
            num_features = X_train.shape[2]
            logger.info(f"Building LSTM model for {num_features} features...")
            # Define model within the try block
            model_lstm = Sequential([
                LSTM(64, input_shape=(SEQ_LEN, num_features), return_sequences=True),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(num_features) # Output layer matches input features
            ])
            model_lstm.compile(optimizer='adam', loss='mse') # Use MSE for autoencoder reconstruction loss

            logger.info(f"Training LSTM model (Epochs: {LSTM_EPOCHS}, Batch: {LSTM_BATCH_SIZE})...")
            history = model_lstm.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
                                     validation_split=LSTM_VALIDATION_SPLIT, shuffle=False, verbose=0)
            logger.info("LSTM training finished.")

            # Plot training history
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
            loss = history.history.get('loss'); val_loss = history.history.get('val_loss')
            plot_hist = False
            if loss: ax_hist.plot(loss, label='Training Loss'); plot_hist = True
            if val_loss: ax_hist.plot(val_loss, label='Validation Loss'); plot_hist = True
            if plot_hist:
                 ax_hist.legend()
                 ax_hist.set_title(f'LSTM Training History - {job_label}'); ax_hist.set_xlabel('Epoch'); ax_hist.set_ylabel('Loss (MSE)')
                 fig_hist.tight_layout(); save_plot(fig_hist, f'{job_label}_lstm_training_history.png')
            else:
                 logger.warning("No loss history recorded for LSTM training plot.")
                 plt.close(fig_hist) # Close the empty figure


            logger.info("LSTM predicting on test set...")
            preds_lstm = model_lstm.predict(X_test)
            # Calculate Mean Absolute Error (MAE) for each time step's reconstruction
            errors_lstm = np.mean(np.abs(preds_lstm - y_test), axis=1)

            if errors_lstm.size > 0:
                # Determine threshold based on percentile of reconstruction errors *on the test set*
                try:
                    threshold_lstm = np.percentile(errors_lstm, ANOMALY_THRESHOLD_PERCENTILE)
                except IndexError: # Can happen if errors_lstm is empty or percentile is invalid
                    logger.error(f"Could not calculate {ANOMALY_THRESHOLD_PERCENTILE}th percentile for LSTM errors. Setting threshold to NaN.")
                    threshold_lstm = np.nan

                if not np.isnan(threshold_lstm):
                    anomalies_lstm_bool = errors_lstm > threshold_lstm

                    # Ensure boolean mask aligns with test timestamps
                    if len(anomalies_lstm_bool) == len(test_timestamps):
                        lstm_anomalies_ts = test_timestamps[anomalies_lstm_bool] # Select timestamps where condition is True
                    else:
                        # This indicates a probable bug in earlier length reconciliation
                        logger.error(f"CRITICAL: LSTM anomaly boolean length ({len(anomalies_lstm_bool)}) mismatches test timestamps ({len(test_timestamps)}). Cannot identify LSTM anomaly timestamps.")
                        lstm_anomalies_ts = pd.DatetimeIndex([], tz='UTC') # Reset to empty

                    logger.info(f"LSTM MAE Threshold (P{ANOMALY_THRESHOLD_PERCENTILE}): {threshold_lstm:.6f}")
                    logger.info(f"LSTM anomalies found in test set: {anomalies_lstm_bool.sum()} ({len(lstm_anomalies_ts)} unique timestamps)")

                    # Plot reconstruction errors
                    fig_err, ax_err = plt.subplots(figsize=(15, 6))
                    ax_err.plot(test_timestamps, errors_lstm, label='LSTM MAE', alpha=0.7, linewidth=1)
                    ax_err.axhline(y=threshold_lstm, color='r', linestyle='--', label=f'Threshold (P{ANOMALY_THRESHOLD_PERCENTILE}) = {threshold_lstm:.4f}')
                    if not lstm_anomalies_ts.empty:
                        # Find indices where anomalies occurred to plot correctly
                        anomaly_indices = np.where(anomalies_lstm_bool)[0]
                        # Ensure indices are within bounds of errors_lstm
                        if anomaly_indices.size > 0 and max(anomaly_indices) < len(errors_lstm):
                             ax_err.scatter(lstm_anomalies_ts, errors_lstm[anomaly_indices], color='red', s=40, label=f'Anomaly ({len(lstm_anomalies_ts)})', marker='x', zorder=5)
                        elif anomaly_indices.size > 0:
                             logger.error("Anomaly indices out of bounds for LSTM errors plot.")

                    ax_err.set_title(f'LSTM Reconstruction Error (MAE) - {job_label}'); ax_err.set_xlabel('Time (UTC)'); ax_err.set_ylabel('Mean Absolute Error')
                    ax_err.legend(); fig_err.tight_layout(); save_plot(fig_err, f'{job_label}_lstm_errors.png')

                    # Push LSTM-specific metrics (only if threshold is valid)
                    lstm_threshold_gauge.labels(job=job_label, model='lstm').set(threshold_lstm)
                    if errors_lstm.size > 0: lstm_mae_gauge.labels(job=job_label).set(errors_lstm[-1]) # Latest MAE
                    if anomalies_lstm_bool.size > 0: lstm_anomaly_latest_marker.labels(job=job_label, model='lstm').set(1 if anomalies_lstm_bool[-1] else 0) # Latest anomaly flag

                else:
                     logger.warning("LSTM threshold is NaN, skipping anomaly detection and metric push for LSTM.")

            else:
                logger.warning("LSTM error calculation resulted in empty array. No threshold or anomalies determined.")
        except Exception as e:
            logger.error(f"Error during LSTM processing: {e}", exc_info=True)
            lstm_anomalies_ts = pd.DatetimeIndex([], tz='UTC') # Ensure reset on error
    elif not TENSORFLOW_AVAILABLE:
         logger.info("Skipping LSTM: TensorFlow/Keras library not available.")
    else:
        # Log reason for skipping if TF is available but data is bad
        if X_train is None or y_train is None: reason = "training data missing"
        elif X_test is None or y_test is None: reason = "test data missing"
        elif test_timestamps.empty: reason = "test timestamps empty"
        else: reason = "unknown data issue"
        logger.warning(f"Skipping LSTM: Invalid data ({reason}).")


    # --- Isolation Forest ---
    # Requires scaled data (specifically y_train, y_test derived from it) and valid test data/timestamps
    if y_train is not None and y_test is not None and not test_timestamps.empty:
         logger.info("--- Running Isolation Forest ---")
         # Train on the target values ('y') from the training split
         iforest_train_data = y_train
         logger.debug(f"IForest training data (y_train) shape: {iforest_train_data.shape}")

         if iforest_train_data is not None and len(iforest_train_data) > 0:
             iforest_test_data = y_test # Test on the actual values from the test set
             contam_val = IFOREST_CONTAMINATION
             # Handle 'auto' or invalid float contamination
             effective_contam = 0.05 # Default effective value if 'auto' or invalid
             if isinstance(contam_val, float) and 0 < contam_val <= 0.5:
                 effective_contam = contam_val
                 logger.info(f"Using Isolation Forest contamination: {effective_contam}")
             else:
                 # If 'auto' or invalid, use the default and log it
                 logger.info(f"Using default Isolation Forest contamination: {effective_contam} (specified: '{contam_val}')")


             logger.info(f"Training Isolation Forest with effective contamination={effective_contam}...")
             try:
                 # Ensure n_estimators is reasonable, especially for small datasets
                 n_est = 100
                 if len(iforest_train_data) < n_est:
                     n_est = max(10, len(iforest_train_data) // 2) # Adjust n_estimators if data is small
                     logger.warning(f"Reduced IForest n_estimators to {n_est} due to small training size ({len(iforest_train_data)}).")

                 iso_forest = IsolationForest(n_estimators=n_est, contamination=effective_contam, random_state=RANDOM_STATE, n_jobs=-1)
                 iso_forest.fit(iforest_train_data)

                 logger.info("IForest predicting anomaly scores on test set (y_test)...")
                 # decision_function returns raw scores (lower is more anomalous).
                 scores_iforest = iso_forest.decision_function(iforest_test_data)

                 if scores_iforest.size > 0:
                     # Calculate threshold based on the percentile of *scores* corresponding to contamination level
                     # Note: Lower scores are more anomalous.
                     score_threshold_iforest = np.percentile(scores_iforest, effective_contam * 100)

                     anomalies_iforest_bool = scores_iforest < score_threshold_iforest
                     # Align with test timestamps
                     if len(anomalies_iforest_bool) == len(test_timestamps):
                         iforest_anomalies_ts = test_timestamps[anomalies_iforest_bool] # Select timestamps where condition is True
                     else:
                         logger.error(f"CRITICAL: IForest anomaly boolean length ({len(anomalies_iforest_bool)}) mismatches test timestamps ({len(test_timestamps)}). Cannot identify IForest anomaly timestamps.")
                         iforest_anomalies_ts = pd.DatetimeIndex([], tz='UTC')

                     logger.info(f"IForest Score Threshold (~P{effective_contam*100:.1f}): {score_threshold_iforest:.6f}")
                     logger.info(f"IForest anomalies found in test set: {anomalies_iforest_bool.sum()} ({len(iforest_anomalies_ts)} unique timestamps)")

                     # Plot IForest scores
                     fig_if, ax_if = plt.subplots(figsize=(15, 6))
                     ax_if.plot(test_timestamps, scores_iforest, label='IF Score', alpha=0.7, linewidth=1)
                     ax_if.axhline(y=score_threshold_iforest, color='r', linestyle='--', label=f'Threshold (~P{effective_contam*100:.1f}) = {score_threshold_iforest:.4f}')
                     if not iforest_anomalies_ts.empty:
                          # Find indices to plot scores correctly
                          anomaly_indices_if = np.where(anomalies_iforest_bool)[0]
                          if anomaly_indices_if.size > 0 and max(anomaly_indices_if) < len(scores_iforest):
                               ax_if.scatter(iforest_anomalies_ts, scores_iforest[anomaly_indices_if], color='red', s=40, label=f'Anomaly ({len(iforest_anomalies_ts)})', marker='o', zorder=5)
                          elif anomaly_indices_if.size > 0:
                               logger.error("Anomaly indices out of bounds for IForest scores plot.")

                     ax_if.set_title(f'Isolation Forest Anomaly Score - {job_label}'); ax_if.set_xlabel('Time (UTC)'); ax_if.set_ylabel('Score (Lower is more anomalous)')
                     ax_if.legend(); fig_if.tight_layout(); save_plot(fig_if, f'{job_label}_iforest_scores.png')
                 else:
                     logger.warning("Isolation Forest score calculation resulted in empty array.")

             except Exception as e:
                logger.error(f"Error during Isolation Forest processing: {e}", exc_info=True)
                iforest_anomalies_ts = pd.DatetimeIndex([], tz='UTC') # Reset on error
         else:
             logger.warning("Skipping Isolation Forest: Training data (y_train) is missing or empty.")
    else:
        # Log reason for skipping
        if y_train is None: reason = "y_train missing"
        elif y_test is None: reason = "y_test missing"
        elif test_timestamps.empty: reason = "test timestamps empty"
        else: reason = "unknown data issue"
        logger.warning(f"Skipping Isolation Forest: Invalid data ({reason}).")


    # --- Prophet ---
    if PROPHET_AVAILABLE and PROPHET_TARGET_METRIC:
        logger.info("--- Running Prophet ---")
        prophet_target_col = PROPHET_TARGET_METRIC
        if prophet_target_col in combined_df.columns:
            # Use the *original* combined (but cleaned) data for Prophet
            prophet_df_orig = combined_df[[prophet_target_col]].copy().dropna()
            if not prophet_df_orig.empty:
                # Prepare DataFrame for Prophet: 'ds' (datetime) and 'y' (value)
                prophet_df_prep = prophet_df_orig.reset_index()
                prophet_df_prep.rename(columns={'timestamp': 'ds', prophet_target_col: 'y'}, inplace=True)

                # Ensure 'ds' is datetime type
                try:
                    prophet_df_prep['ds'] = pd.to_datetime(prophet_df_prep['ds']) # ds should inherit UTC tz
                    if prophet_df_prep['ds'].dt.tz is None:
                         logger.warning("Prophet 'ds' column became timezone-naive. Assuming UTC.")
                         prophet_df_prep['ds'] = prophet_df_prep['ds'].dt.tz_localize('UTC')
                    elif prophet_df_prep['ds'].dt.tz != timezone.utc:
                        logger.warning("Prophet 'ds' column was not UTC after conversion. Forcing UTC.")
                        prophet_df_prep['ds'] = prophet_df_prep['ds'].dt.tz_convert('UTC')
                except Exception as e:
                     logger.error(f"Failed to convert or validate 'ds' column for Prophet: {e}", exc_info=True)
                     # Attempt to continue might fail later

                # Handle potential duplicate timestamps ('ds' column)
                dupes_ds = prophet_df_prep['ds'].duplicated().sum()
                if dupes_ds > 0:
                    logger.warning(f"Prophet input has {dupes_ds} duplicate timestamps in 'ds'. Averaging 'y' values for duplicates.")
                    # Keep first timestamp for each group, average y
                    prophet_df_prep = prophet_df_prep.groupby('ds', as_index=False).agg({'y': 'mean'})
                    # Note: Index is likely reset by groupby here

                # Drop any NaNs that might have appeared in 'y' during grouping
                prophet_df_prep = prophet_df_prep.dropna(subset=['y']).reset_index(drop=True)
                # ^^ Index should be unique default range index here

                # Prophet requires at least 2 data points
                if len(prophet_df_prep) >= 2:
                    prophet_df_fit = prophet_df_prep.copy()
                    # Prophet works best with timezone-naive timestamps internally for fitting
                    original_tz = None
                    try:
                        if prophet_df_fit['ds'].dt.tz is not None:
                            original_tz = prophet_df_fit['ds'].dt.tz # Store original TZ
                            prophet_df_fit['ds'] = prophet_df_fit['ds'].dt.tz_localize(None)
                            logger.debug(f"Removed timezone ({original_tz}) from 'ds' column for Prophet fitting.")
                        else:
                             logger.debug("'ds' column is already timezone-naive for Prophet fitting.")
                    except Exception as tz_err:
                        logger.error(f"Error adjusting timezone for Prophet 'ds' column: {tz_err}. Proceeding.", exc_info=True)


                    try:
                        # <<< FIX Integration: Explicitly check and reset index before fitting >>>
                        idx_dupes_before_fit = prophet_df_fit.index.duplicated().sum()
                        if idx_dupes_before_fit > 0:
                            logger.warning(f"Found {idx_dupes_before_fit} duplicate indices in prophet_df_fit right before fit. Resetting index.")
                            prophet_df_fit = prophet_df_fit.reset_index(drop=True)
                        # else:
                        #     logger.debug("Prophet df_fit index is unique before fitting.") # Optional: uncomment for verbose debugging

                        # Optional Sanity Check: Verify 'ds' uniqueness again just before fit
                        ds_dupes_before_fit = prophet_df_fit['ds'].duplicated().sum()
                        if ds_dupes_before_fit > 0:
                             # This shouldn't happen after the groupby, but is a critical warning if it does
                             logger.error(f"CRITICAL WARNING: Duplicate 'ds' values ({ds_dupes_before_fit}) found in prophet_df_fit immediately before fit. Prophet may fail or yield incorrect results.")


                        logger.info(f"Fitting Prophet model for '{prophet_target_col}' on {len(prophet_df_fit)} points...")
                        # Adjust seasonality based on data duration
                        data_duration_days = 0
                        if not prophet_df_fit.empty:
                            data_duration_days = (prophet_df_fit['ds'].max() - prophet_df_fit['ds'].min()).days

                        use_yearly = 'auto' if data_duration_days > 365 * 1.5 else False
                        use_weekly = 'auto' if data_duration_days > 14 else False
                        use_daily = 'auto' if data_duration_days > 2 else False
                        logger.debug(f"Prophet seasonality settings: yearly={use_yearly}, weekly={use_weekly}, daily={use_daily}")

                        prophet_model = Prophet(interval_width=PROPHET_INTERVAL_WIDTH,
                                                yearly_seasonality=use_yearly,
                                                weekly_seasonality=use_weekly,
                                                daily_seasonality=use_daily)

                        # --- This is the line that previously failed ---
                        prophet_model.fit(prophet_df_fit)
                        # --- End of potentially failing line ---

                        logger.info("Prophet making predictions (forecasting on training data points)...")
                        # Predict on the same dates we trained on to get uncertainty intervals
                        # 'future_df' here is actually the historical data frame for prediction
                        forecast = prophet_model.predict(prophet_df_fit[['ds']]) # Predict needs 'ds' column

                        # Merge forecast results back with the original prepared data
                        # Merge on 'ds'. Ensure 'ds' in forecast is compatible. Prophet predict usually returns naive ds.
                        if forecast['ds'].dt.tz is not None:
                             logger.warning("Prophet forecast 'ds' column has timezone. Removing for merge.")
                             forecast['ds'] = forecast['ds'].dt.tz_localize(None)

                        # Merge original (potentially tz-naive fit df) with forecast (naive ds)
                        results_prophet = pd.merge(
                            prophet_df_fit, # Contains naive 'ds' and 'y'
                            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                            on='ds',
                            how='left' # Keep all original points, match forecast where possible
                        )

                        # Add original timezone back if it existed, crucial for comparing with test_timestamps
                        if original_tz is not None:
                             logger.debug(f"Adding original timezone ({original_tz}) back to Prophet results 'ds'.")
                             # Convert to UTC for consistency
                             results_prophet['ds'] = results_prophet['ds'].dt.tz_localize(original_tz).dt.tz_convert('UTC')
                        elif results_prophet['ds'].dt.tz is None:
                             # If original was naive AND we need to compare to UTC test timestamps, assume UTC was intended
                             logger.warning("Original Prophet 'ds' was naive. Assuming UTC for results.")
                             results_prophet['ds'] = results_prophet['ds'].dt.tz_localize('UTC')
                        elif results_prophet['ds'].dt.tz != timezone.utc:
                             # If it has TZ but not UTC, convert
                             logger.warning(f"Prophet results 'ds' had timezone {results_prophet['ds'].dt.tz}. Converting to UTC.")
                             results_prophet['ds'] = results_prophet['ds'].dt.tz_convert('UTC')


                        # Identify anomalies: points where actual 'y' is outside the forecast interval [yhat_lower, yhat_upper]
                        # Handle potential NaNs in forecast bounds
                        results_prophet['anomaly'] = (
                            (results_prophet['y'] < results_prophet['yhat_lower']) | \
                            (results_prophet['y'] > results_prophet['yhat_upper'])
                        ).fillna(False) # Treat comparison with NaN bounds as not anomalous


                        # Filter anomalies to only those within the designated test period
                        if not test_timestamps.empty and not results_prophet.empty:
                             # Ensure results_prophet['ds'] is UTC for comparison
                             if results_prophet['ds'].dt.tz != timezone.utc:
                                 logger.warning("Converting Prophet results 'ds' timezone to UTC for filtering.")
                                 results_prophet['ds'] = results_prophet['ds'].dt.tz_convert('UTC')

                             # Perform the filtering using boolean indexing
                             is_in_test_period = results_prophet['ds'].isin(test_timestamps)
                             prophet_anomalies_in_test_period_df = results_prophet[is_in_test_period & results_prophet['anomaly']]

                             # Extract the timestamps (already UTC)
                             prophet_anomalies_ts = pd.DatetimeIndex(prophet_anomalies_in_test_period_df['ds']).tz_convert('UTC')
                             logger.info(f"Prophet anomalies found within test period: {len(prophet_anomalies_ts)}")
                        else:
                             logger.warning("Cannot filter Prophet anomalies to test period: Test timestamps empty or Prophet results empty.")
                             prophet_anomalies_ts = pd.DatetimeIndex([], tz='UTC')

                        # Plot Prophet forecast (using Prophet's plotting functions)
                        try:
                            fig_prophet = prophet_model.plot(forecast) # forecast df uses naive time
                            ax_prophet = fig_prophet.gca()
                            # Overlay detected anomalies *in the test period* on the plot
                            if not prophet_anomalies_ts.empty and not prophet_anomalies_in_test_period_df.empty:
                                # Get the original 'y' values for the anomalous points
                                y_values_for_plot = prophet_anomalies_in_test_period_df['y']
                                # Convert anomaly timestamps to naive for plotting on Prophet's naive axis
                                plot_ts_naive = prophet_anomalies_ts.tz_localize(None)

                                ax_prophet.scatter(plot_ts_naive.to_pydatetime(),
                                                   y_values_for_plot,
                                                   color='red', s=40, label=f'Anomaly in Test ({len(prophet_anomalies_ts)})', marker='x', zorder=10)
                                ax_prophet.legend()

                            ax_prophet.set_title(f'Prophet Forecast & Anomalies ({prophet_target_col}) - {job_label}')
                            ax_prophet.set_xlabel('Time (Prophet Internal - Naive)'); ax_prophet.set_ylabel(prophet_target_col)
                            fig_prophet.tight_layout()
                            save_plot(fig_prophet, f'{job_label}_prophet_forecast.png')
                        except Exception as plot_err:
                            logger.error(f"Error plotting Prophet forecast: {plot_err}", exc_info=True)


                        # Plot Prophet components
                        try:
                           fig_comp = prophet_model.plot_components(forecast)
                           save_plot(fig_comp, f'{job_label}_prophet_components.png')
                        except ValueError as ve:
                            # Handle cases where components can't be plotted (e.g., insufficient data for seasonality)
                            logger.warning(f"Could not plot Prophet components: {ve}")
                        except Exception as plot_comp_err:
                           logger.warning(f"Unexpected error plotting Prophet components: {plot_comp_err}", exc_info=True)

                    except ValueError as ve:
                         # Catch common Prophet ValueErrors during fit/predict
                         # <<< Check if the error is the specific one we aimed to fix >>>
                         if "cannot reindex on an axis with duplicate labels" in str(ve):
                              logger.error(f"Prophet ValueError (Duplicate Index Labels): {ve}. This occurred despite attempting to reset the index.", exc_info=True)
                         else:
                              logger.error(f"Prophet ValueError during processing: {ve}", exc_info=True)
                         prophet_anomalies_ts = pd.DatetimeIndex([], tz='UTC')
                    except Exception as e:
                        logger.error(f"Unexpected error during Prophet processing for '{prophet_target_col}': {e}", exc_info=True)
                        prophet_anomalies_ts = pd.DatetimeIndex([], tz='UTC') # Reset on error
                else:
                    logger.warning(f"Skipping Prophet for '{prophet_target_col}': Not enough data points ({len(prophet_df_prep)}) after preprocessing (need >= 2).")
            else:
                logger.warning(f"Skipping Prophet for '{prophet_target_col}': No data after initial selection and dropna.")
        else:
            logger.warning(f"Skipping Prophet: Target metric '{prophet_target_col}' not found in combined DataFrame columns ({combined_df.columns}).")
    elif not PROPHET_AVAILABLE:
         logger.info("Skipping Prophet: Library not available.") # Info level sufficient if expected
    elif not PROPHET_TARGET_METRIC:
         logger.warning("Skipping Prophet: Target metric not configured or found.")


    # 5. Consolidate Anomalies, Evaluate Alert Rules & Push Metrics
    logger.info("Step 5: Consolidating anomalies, evaluating alert rules, and pushing metrics...")

    # --- Consolidate all detected anomalies from the test period ---
    combined_anomalies_cycle = defaultdict(list)
    logger.debug(f"Consolidating raw anomalies: LSTM={len(lstm_anomalies_ts)}, IF={len(iforest_anomalies_ts)}, Prophet={len(prophet_anomalies_ts)}")
    # Ensure all timestamps added are UTC and handle potential NaT values gracefully
    # Use .dropna() after conversion to handle any NaT results
    for ts in pd.to_datetime(lstm_anomalies_ts, utc=True).dropna(): combined_anomalies_cycle[ts].append('lstm')
    for ts in pd.to_datetime(iforest_anomalies_ts, utc=True).dropna(): combined_anomalies_cycle[ts].append('iforest')
    for ts in pd.to_datetime(prophet_anomalies_ts, utc=True).dropna(): combined_anomalies_cycle[ts].append('prophet')


    combined_anomalies_cycle_dict = dict(combined_anomalies_cycle)
    num_unique_anomalies = len(combined_anomalies_cycle_dict)
    logger.info(f"Total unique anomaly timestamps identified by any model in test period: {num_unique_anomalies}")

    # --- Set Prometheus markers for *all* detected anomalies ---
    # It's useful to see all detected points in Prometheus, even if not alerted
    set_combined_anomaly_metrics(combined_anomaly_marker, combined_anomalies_cycle_dict, job_label)

    # --- Evaluate Significance for Email Alerting (Consensus/Persistence) ---
    logger.info("Step 5b: Checking for significant anomalies based on consensus/persistence...")
    significant_anomalies_candidates = identify_significant_anomalies_combined(
        combined_anomalies=combined_anomalies_cycle_dict,
        test_timestamps=test_timestamps, # Pass the actual test timestamps (should be UTC)
        min_models=MIN_MODELS_FOR_ALERT,
        enable_persistence=ENABLE_PERSISTENCE_ALERT,
        min_consecutive=MIN_CONSECUTIVE_ANOMALIES,
        window_size=PERSISTENCE_WINDOW_SIZE,
        min_in_window=MIN_ANOMALIES_IN_WINDOW
    )

    # Filter significant anomalies for recency
    anomalies_to_alert = {}
    if significant_anomalies_candidates:
        now_utc = datetime.now(timezone.utc)
        alert_time_threshold = now_utc - timedelta(minutes=ALERT_RECENCY_MINUTES)
        logger.info(f"Filtering significant anomalies: Keeping only those since {alert_time_threshold.strftime('%Y-%m-%d %H:%M:%S %Z')}...")

        for ts, models in significant_anomalies_candidates.items():
             # Ensure timestamp is timezone-aware UTC for comparison
             # identify_significant_anomalies_combined should return UTC keys
             if ts >= alert_time_threshold:
                anomalies_to_alert[ts] = models # Keep the original timestamp object

        num_filtered_out = len(significant_anomalies_candidates) - len(anomalies_to_alert)
        if num_filtered_out > 0:
            logger.info(f"Filtered out {num_filtered_out} significant anomalies older than {ALERT_RECENCY_MINUTES} minutes.")
        logger.info(f"Total anomalies meeting significance AND recency criteria: {len(anomalies_to_alert)}")
    else:
        logger.info("No significant anomalies found based on consensus/persistence rules.")


    # --- Send Email ONLY if Recent Significant Anomalies Found ---
    # Pass the filtered dictionary `anomalies_to_alert`
    send_anomaly_email(anomalies_to_alert, job_label)


    # --- Push all collected metrics (Gauges set earlier + anomaly markers) ---
    logger.info(f"Pushing all collected metrics to Pushgateway ({PUSHGATEWAY_URL}) for job '{job_label}'...")
    try:
        push_to_gateway(PUSHGATEWAY_URL, job=job_label, registry=registry, timeout=30) # Add timeout
        logger.info("Successfully pushed metrics.")
    except (requests.exceptions.ConnectionError, socket.gaierror, requests.exceptions.Timeout, ConnectionRefusedError) as conn_err:
         # Log specific connection errors
         logger.error(f"Network/Connection Error pushing metrics to Pushgateway {PUSHGATEWAY_URL}: {conn_err}")
    except Exception as e:
        # Catch other errors during push, e.g., authentication, invalid data
        logger.error(f"Failed to push metrics to Pushgateway {PUSHGATEWAY_URL}: {e}", exc_info=True)


    # 6. Log Cycle Summary
    end_cycle_time = datetime.now(timezone.utc)
    duration = end_cycle_time - start_cycle_time
    logger.info("--- Anomaly Detection Cycle Summary ---")
    if not test_timestamps.empty:
        logger.info(f"Data Fetch Range: ~{start_fetch_time.isoformat()} to {end_fetch_time.isoformat()}")
        logger.info(f"Test Period Analysed: {test_timestamps.min().isoformat()} to {test_timestamps.max().isoformat()}")
    else:
        logger.info("Test Period: N/A (Split failed or insufficient data)")
    logger.info(f"Anomalies Found (Raw): LSTM={len(lstm_anomalies_ts)}, IF={len(iforest_anomalies_ts)}, Prophet={len(prophet_anomalies_ts)}")
    logger.info(f"Total Unique Anomaly Timestamps Found (Any Model): {num_unique_anomalies}")
    logger.info(f"Significant Anomalies (Consensus/Persistence): {len(significant_anomalies_candidates)}")
    logger.info(f"Recent Significant Anomalies (Last {ALERT_RECENCY_MINUTES} min, Triggering Alert): {len(anomalies_to_alert)}") # Updated log
    logger.info(f"Cycle Duration: {duration}")
    logger.info(f"--- Cycle Finished for Job: {job_label} ---")


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Script starting execution.")
    # Basic check for essential email config if alerting is intended
    if not EMAIL_RECIPIENTS:
        logger.error("EMAIL_RECIPIENTS is not set or empty. Email alerting will be disabled.")
    else:
        # Log recipient only if it's not the hardcoded default (or always log if needed)
        logger.info(f"Email alerts configured to be sent to: {EMAIL_RECIPIENTS}")
        if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD]):
             logger.warning("One or more SMTP configuration variables (HOST, PORT, USER, PASSWORD) are missing or empty. Email sending may fail.")

    try:
        run_anomaly_detection(job_label=JOB_NAME)
    except Exception as main_err:
        logger.critical(f"Unhandled exception occurred in main execution block: {main_err}", exc_info=True)
        # Attempt to send a basic failure email here if possible and configured
        try:
             # Simple email function for critical failure
             def send_failure_email(error_msg, job_name_fail):
                 if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_RECIPIENTS]):
                      logger.error("Cannot send failure email: Email configuration incomplete.")
                      return
                 recipient_list = [e.strip() for e in EMAIL_RECIPIENTS.split(',') if e.strip()]
                 if not recipient_list:
                     logger.error("Cannot send failure email: No valid recipients.")
                     return

                 msg = EmailMessage()
                 msg['Subject'] = f"CRITICAL FAILURE: Anomaly Detector Job '{job_name_fail}'"
                 msg['From'] = EMAIL_SENDER
                 msg['To'] = ", ".join(recipient_list)
                 msg.set_content(f"The anomaly detection script '{job_name_fail}' failed critically and stopped execution.\n\nPlease check the logs.\n\nError:\n{error_msg}\n\nTimestamp (UTC): {datetime.now(timezone.utc)}")

                 logger.info(f"Attempting to send critical failure email notification for job '{job_name_fail}'...")
                 # Use same SMTP logic as regular alerts
                 if SMTP_PORT == 465:
                      with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:
                           server.login(SMTP_USER, SMTP_PASSWORD)
                           server.send_message(msg)
                 else:
                      with smtplib.SMTP(HOST, SMTP_PORT, timeout=30) as server:
                           server.starttls()
                           server.login(SMTP_USER, SMTP_PASSWORD)
                           server.send_message(msg)
                 logger.info("Sent critical failure email notification.")

             send_failure_email(str(main_err), JOB_NAME)
        except Exception as email_fail_err:
             # Log error if sending the failure email *also* fails
             logger.error(f"Failed to send critical failure email notification: {email_fail_err}", exc_info=True)

    finally:
        logger.info("Script execution finished.")