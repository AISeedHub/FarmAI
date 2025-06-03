# Time Series Text Generator for LLM-based Forecasting

This module provides functionality to generate text descriptions of time series data for use with Large Language Models (LLMs) in forecasting tasks.

## Overview

The `text_generator.py` module converts numerical time series data into descriptive text prompts that include statistical information, trend analysis, and seasonality detection. These prompts can be used as input to LLMs to leverage their capabilities for time series forecasting.

## Features

- Generate text descriptions of time series data with statistical summaries
- Detect trends (increasing, decreasing, stationary) in time series
- Detect seasonality patterns
- Handle multiple variables and sequences
- Process both numpy arrays and pandas DataFrames
- Handle missing values

## Functions

### `detect_trend(series)`

Detects if a time series has an increasing, decreasing, or stationary trend.

**Parameters:**
- `series` (numpy.ndarray): The time series data

**Returns:**
- `str`: 'increasing', 'decreasing', or 'stationary'

### `detect_seasonality(series, freq=None)`

Detects if a time series has seasonality.

**Parameters:**
- `series` (numpy.ndarray): The time series data
- `freq` (int, optional): The frequency of the seasonality to check

**Returns:**
- `str`: Description of seasonality

### `generate_prompt(series, variable_name="the variable", n_forecast=None)`

Generates a text prompt for LLM-based forecasting based on time series data.

**Parameters:**
- `series` (numpy.ndarray): The time series data
- `variable_name` (str): Name of the variable
- `n_forecast` (int, optional): Number of values to forecast

**Returns:**
- `str`: Formatted prompt for LLM

### `batch_generate_prompts(data, variables=None, seq_len=96, pred_len=None)`

Generates prompts for multiple variables and/or multiple sequences.

**Parameters:**
- `data` (numpy.ndarray or pandas.DataFrame): The time series data
- `variables` (list, optional): List of variable names
- `seq_len` (int): Length of each sequence
- `pred_len` (int, optional): Number of values to forecast

**Returns:**
- `list`: List of prompts

## Example Usage

### Basic Usage

```python
import pandas as pd
from utils.text_generator import generate_prompt

# Load your time series data
df = pd.read_csv('your_data.csv')

# Generate a prompt for a single variable
co2_data = df['CO2'].values[:96]  # Use 96 time steps
prompt = generate_prompt(co2_data, variable_name="CO2", n_forecast=24)
print(prompt)
```

### Multiple Variables

```python
from utils.text_generator import batch_generate_prompts

# Generate prompts for multiple variables
variables = ['CO2', 'Temperature', 'Humidity']
prompts = batch_generate_prompts(df[variables], variables=variables, seq_len=96, pred_len=24)

# Print the first prompt
print(prompts[0])
```

### Integration with LLM

```python
# Generate prompts
prompts = batch_generate_prompts(df, seq_len=96, pred_len=24)

# Send prompts to LLM (example with a hypothetical LLM API)
for prompt in prompts:
    # Call your LLM API
    llm_response = call_llm_api(prompt)
    
    # Parse the response to get forecasted values
    forecasted_values = parse_llm_response(llm_response)
    
    # Use the forecasted values
    # ...
```

## Handling Missing Values

Before generating prompts, it's recommended to handle any missing values in your data:

```python
# Fill missing values
df_filled = df.fillna(method='ffill')  # Forward fill

# Generate prompt
prompt = generate_prompt(df_filled['CO2'].values[:96], variable_name="CO2", n_forecast=24)
```

## Demo Script

A demonstration script `demo_text_generator.py` is provided in the project root directory. Run it to see examples of how to use the text generation module:

```
python demo_text_generator.py
```

## Dependencies

- numpy
- pandas
- statsmodels (for trend and seasonality detection)

## Notes

- The module automatically handles long sequences by showing only the first and last 10 values in the prompt.
- Trend detection uses both linear regression and the Augmented Dickey-Fuller test.
- Seasonality detection attempts to automatically identify the appropriate frequency.