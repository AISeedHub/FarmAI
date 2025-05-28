import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Try to import statsmodels, but provide fallback if not available
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Using simplified trend and seasonality detection.")
    print("For better results, install statsmodels: pip install statsmodels")

def detect_trend(series):
    """
    Detect if a time series has an increasing, decreasing, or stationary trend.

    Args:
        series (numpy.ndarray): The time series data

    Returns:
        str: 'increasing', 'decreasing', or 'stationary'
    """
    # Simple linear regression to detect trend
    x = np.arange(len(series))
    slope = np.polyfit(x, series, 1)[0]

    # If statsmodels is available, use Augmented Dickey-Fuller test for stationarity
    is_stationary = False
    if STATSMODELS_AVAILABLE:
        try:
            result = adfuller(series)
            p_value = result[1]
            is_stationary = p_value < 0.05
        except:
            pass
    else:
        # Simple heuristic for stationarity when statsmodels is not available
        # Calculate if the series stays within a certain range of its mean
        mean = np.mean(series)
        std = np.std(series)
        is_within_range = np.all(np.abs(series - mean) < 2 * std)
        is_stationary = is_within_range and abs(slope) < 0.005

    if is_stationary:
        return "stationary"
    elif slope > 0.01:  # Threshold can be adjusted
        return "increasing"
    elif slope < -0.01:  # Threshold can be adjusted
        return "decreasing"
    else:
        return "stationary"

def detect_seasonality(series, freq=None):
    """
    Detect if a time series has seasonality.

    Args:
        series (numpy.ndarray): The time series data
        freq (int, optional): The frequency of the seasonality to check

    Returns:
        str: Description of seasonality
    """
    # If series is too short, return no seasonality
    if len(series) < 4:
        return "no clear seasonality"

    # Determine frequency if not provided
    if freq is None:
        # Try to automatically detect frequency
        if len(series) >= 24:
            freq = 24  # Daily seasonality (if hourly data)
        elif len(series) >= 7:
            freq = 7   # Weekly seasonality
        else:
            freq = 2   # Minimum required

    # If statsmodels is available, use seasonal decomposition
    if STATSMODELS_AVAILABLE and len(series) >= 2*freq:
        try:
            result = seasonal_decompose(series, model='additive', period=freq)
            seasonal_strength = np.std(result.seasonal) / np.std(result.resid)

            if seasonal_strength > 0.5:  # Threshold can be adjusted
                if freq == 24:
                    return "shows daily seasonality"
                elif freq == 7:
                    return "shows weekly seasonality"
                else:
                    return f"shows seasonality with period {freq}"
        except:
            pass
    else:
        # Simple heuristic for seasonality when statsmodels is not available
        # Check for repeating patterns by comparing segments of the series
        if len(series) >= 2*freq:
            # Split the series into segments of length freq
            segments = [series[i:i+freq] for i in range(0, len(series)-freq, freq)]

            if len(segments) >= 2:
                # Calculate correlation between adjacent segments
                correlations = []
                for i in range(len(segments)-1):
                    seg1 = segments[i][:min(len(segments[i]), len(segments[i+1]))]
                    seg2 = segments[i+1][:min(len(segments[i]), len(segments[i+1]))]
                    if len(seg1) > 0 and len(seg2) > 0:
                        corr = np.corrcoef(seg1, seg2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

                # If average correlation is high, there's likely seasonality
                if correlations and np.mean(correlations) > 0.5:
                    if freq == 24:
                        return "shows daily seasonality"
                    elif freq == 7:
                        return "shows weekly seasonality"
                    else:
                        return f"shows seasonality with period {freq}"

    return "no clear seasonality"

def generate_prompt(series, variable_name="the variable", n_forecast=None):
    """
    Generate a text prompt for LLM-based forecasting based on time series data.

    Args:
        series (numpy.ndarray): The time series data
        variable_name (str): Name of the variable
        n_forecast (int, optional): Number of values to forecast

    Returns:
        str: Formatted prompt for LLM
    """
    # Calculate statistics
    mean_value = np.mean(series)
    std_value = np.std(series)
    min_value = np.min(series)
    max_value = np.max(series)

    # Detect trend and seasonality
    trend = detect_trend(series)
    seasonality = detect_seasonality(series)

    # Format series values (limit to first 10 and last 10 if very long)
    # if len(series) > 20:
    #     series_str = ', '.join([f"{x:.1f}" for x in series[:10]])
    #     series_str += ', ...'
    #     series_str += ', '.join([f"{x:.1f}" for x in series[-10:]])
    # else:
    series_str = ', '.join([f"{x:.1f}" for x in series])

    # Create the prompt
    prompt = f"""The following is a time series of {variable_name} measured at regular intervals:
[{series_str}]

Statistical summary:
- Mean: {mean_value:.2f}
- Standard deviation: {std_value:.2f}
- Minimum: {min_value:.2f}
- Maximum: {max_value:.2f}
- Trend: {trend}
- Seasonality: {seasonality}
"""

    if n_forecast is not None:
        prompt += f"\nBased on the above data and its statistical characteristics, forecast the next {n_forecast} values."

    return prompt

def batch_generate_prompts(data, variables=None, seq_len=96, pred_len=None):
    """
    Generate prompts for multiple variables and/or multiple sequences.

    Args:
        data (numpy.ndarray or pandas.DataFrame): The time series data
                If numpy array: shape should be [samples, variables]
                If DataFrame: each column is a variable
        variables (list, optional): List of variable names
        seq_len (int): Length of each sequence
        pred_len (int, optional): Number of values to forecast

    Returns:
        list: List of prompts
    """
    # Convert to DataFrame if numpy array
    if isinstance(data, np.ndarray):
        if variables is None:
            variables = [f"variable_{i+1}" for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=variables)

    # If variables not specified, use all columns
    if variables is None:
        variables = data.columns.tolist()

    prompts = []

    # Generate prompts for each variable
    for var in variables:
        if var in data.columns:
            series = data[var].values

            # If series is longer than seq_len, create multiple sequences
            if len(series) > seq_len:
                for i in range(0, len(series) - seq_len + 1, seq_len):
                    seq = series[i:i+seq_len]
                    prompt = generate_prompt(seq, variable_name=var, n_forecast=pred_len)
                    prompts.append(prompt)
            else:
                prompt = generate_prompt(series, variable_name=var, n_forecast=pred_len)
                prompts.append(prompt)

    return prompts

# Example usage:
# df = pd.read_csv('path_to_data.csv')
# prompts = batch_generate_prompts(df, seq_len=96, pred_len=24)
# for prompt in prompts[:2]:  # Print first two prompts
#     print(prompt)
#     print('-' * 80)
