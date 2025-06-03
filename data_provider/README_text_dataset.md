# Text Dataset Loader for LLM-based Forecasting

This document explains how to use the `Dataset_TextPrompt` class to load and work with text datasets generated for LLM-based forecasting.

## Overview

The `Dataset_TextPrompt` class is designed to load text datasets that contain text descriptions of time series data. These text descriptions can be used as input to Large Language Models (LLMs) for forecasting tasks. The class maintains the same interface as the `Dataset_Custom` class, making it easy to integrate with existing code.

## Features

- Load text datasets with minimal code changes
- Support for both compact and full versions of the dataset
- Automatic train/validation/test splitting (60%/20%/20%)
- Compatible with existing data loading infrastructure
- Maintains the same interface as `Dataset_Custom`

## Usage

### Basic Usage

```python
from data_provider.data_loader import Dataset_TextPrompt

# Create a dataset instance
dataset = Dataset_TextPrompt(
    root_path='./dataset/',
    data_path='text_dataset_compact_in96_out96.csv',
    flag='train',
    size=[96, 48, 96],  # [seq_len, label_len, pred_len]
    target='CO2'  # Target variable (or 'all' for all variables)
)

# Access data using the same interface as Dataset_Custom
for i in range(len(dataset)):
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[i]
    # Use the data...

# Access text descriptions directly
text = dataset.text_descriptions[0]
variable = dataset.variables[0]
sequence_id = dataset.sequence_ids[0]
```

### Creating Data Loaders

```python
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_TextPrompt

# Create dataset instances for train, validation, and test sets
train_data = Dataset_TextPrompt(
    root_path='./dataset/',
    data_path='text_dataset_compact_in96_out96.csv',
    flag='train',
    size=[96, 48, 96],
    target='CO2'
)

# Create a data loader
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

# Use the data loader with existing models
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    # Your model training code...
    pass
```

## Dataset Format

The `Dataset_TextPrompt` class can work with two formats of the text dataset:

1. **Full Version**: Contains all columns including `sequence_id`, `variable`, `text_description`, `input_values`, and `output_values`.
2. **Compact Version**: Contains only `sequence_id`, `variable`, and `text_description`.

If you use the compact version, the class will attempt to load the full version to get the input and output values. If the full version is not available, it will create dummy data for compatibility.

## Parameters

- `root_path` (str): Path to the dataset directory
- `data_path` (str): Name of the dataset file
- `flag` (str): 'train', 'test', or 'val'
- `size` (list): [seq_len, label_len, pred_len]
- `target` (str): Target variable name (or 'all' for all variables)
- `features` (str): 'S' or 'M' (single or multiple features)
- `scale` (bool): Whether to scale the data (not used in this class)
- `timeenc` (int): Time encoding method (not used in this class)
- `freq` (str): Frequency of the data (not used in this class)
- `cols` (list): List of columns to use (not used in this class)

## Example Script

An example script is provided in `examples/text_dataset_example.py` to demonstrate how to use the `Dataset_TextPrompt` class. Run it to see how to load and work with the text dataset:

```
python examples/text_dataset_example.py
```

## Integration with Existing Code

The `Dataset_TextPrompt` class is designed to be a drop-in replacement for `Dataset_Custom`. You can use it with your existing code by simply changing the dataset class:

```python
# Before
from data_provider.data_loader import Dataset_Custom
dataset = Dataset_Custom(...)

# After
from data_provider.data_loader import Dataset_TextPrompt
dataset = Dataset_TextPrompt(...)
```

The rest of your code can remain unchanged, as the interface is the same.