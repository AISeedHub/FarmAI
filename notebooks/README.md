# Time Series Text Generation Dataset Builder

This directory contains a Jupyter notebook for processing time series data and generating text descriptions for LLM-based forecasting tasks.

## Overview

The `text_generation_dataset.ipynb` notebook processes the original time series dataset to create a new dataset that includes text descriptions generated from the time series data. These text descriptions can be used as input to Large Language Models (LLMs) for forecasting tasks.

## Features

- Load and preprocess time series data
- Split data into sequences of specified input and output lengths
- Generate text descriptions with statistical information for each sequence
- Visualize sample sequences
- Save the processed dataset to CSV files
- Provide examples of how to use the dataset with an LLM

## How to Use

1. Open the `text_generation_dataset.ipynb` notebook in Jupyter or your preferred notebook environment.
2. Run the cells in order to process the dataset and generate text descriptions.
3. Adjust the parameters as needed:
   - `input_len`: Length of input sequence (default: 96)
   - `output_len`: Length of output sequence to predict (default: 96)
   - `variables`: List of variables to include in the dataset

## Output Files

The notebook generates two CSV files:

1. `text_dataset_in{input_len}_out{output_len}.csv`: Complete dataset with sequences and text descriptions
   - Contains sequence IDs, variable names, text descriptions, input values, and output values
   - Useful for detailed analysis and visualization

2. `text_dataset_compact_in{input_len}_out{output_len}.csv`: Compact version with essential information
   - Contains only sequence IDs, variable names, and text descriptions
   - Ideal for use with LLMs

## Example Usage with LLMs

The notebook includes an example of how to use the generated dataset with an LLM:

```python
# 1. Load the dataset
text_dataset = pd.read_csv('dataset/text_dataset_compact_in96_out96.csv')

# 2. Select a sequence and variable
sequence_id = 0
variable = 'CO2'
prompt = text_dataset[(text_dataset['sequence_id'] == sequence_id) & 
                     (text_dataset['variable'] == variable)]['text_description'].values[0]

# 3. Send the prompt to an LLM
llm_response = call_llm_api(prompt)  # Replace with actual LLM API call

# 4. Parse the response to extract forecasted values
forecasted_values = parse_llm_response(llm_response)  # Implement parsing logic

# 5. Use the forecasted values
print(f"Forecasted values: {forecasted_values}")
```

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- utils.text_generator module (from the project)

## Notes

- The notebook handles missing values and extreme values in the dataset.
- For Sun columns, values greater than 1000 are considered extreme and are replaced.
- The text descriptions include statistical information, trend analysis, and seasonality detection.