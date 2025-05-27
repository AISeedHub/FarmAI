# LLMPatchTST: Time Series Forecasting with LLM and PatchTST

This document explains how to use the `LLMPatchTST` model, which combines Large Language Model (LLM) embeddings with time series data using cross-attention for improved forecasting.

## Overview

The `LLMPatchTST` model is designed to leverage both textual descriptions and time series data for forecasting tasks. It consists of:

1. **LLM Embedding Component**: Processes text descriptions using a pre-trained LLM (e.g., Llama 3)
2. **Time Series Encoder**: Uses PatchTST architecture to process time series data
3. **Cross-Attention Fusion**: Combines the outputs of the two branches using cross-attention
4. **Forecasting Head**: Makes predictions based on the fused features

## Architecture

```
                  ┌─────────────────┐
                  │ Text Description │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   LLM Encoder   │
                  └────────┬────────┘
                           │
                           ▼
┌─────────────┐    ┌─────────────────┐
│ Time Series │    │  LLM Embeddings  │
└──────┬──────┘    └────────┬────────┘
       │                    │
       ▼                    │
┌─────────────┐             │
│ PatchTST    │             │
│  Encoder    │             │
└──────┬──────┘             │
       │                    │
       ▼                    ▼
       │    ┌─────────────────┐
       └───►│ Cross-Attention │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Forecasting Head│
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │    Prediction   │
            └─────────────────┘
```

## Features

- **Dual-Input Processing**: Handles both time series data and text descriptions
- **LLM Integration**: Uses pre-trained LLMs for text embedding (with fallback to random embeddings if LLM is unavailable)
- **Cross-Attention Fusion**: Uses queries from time series and keys/values from LLM embeddings
- **Compatibility**: Maintains the same interface as other models in the framework
- **Flexibility**: Works with various LLM models from Hugging Face

## Requirements

- PyTorch
- Transformers (Hugging Face)
- Access to pre-trained LLM models (optional, will use random embeddings if unavailable)

## Usage

### Basic Usage

```bash
python experiment.py --model LLMPatchTST --data farm --root_path ./dataset/ --data_path text_dataset_compact_in96_out96.csv --features M --seq_len 96 --label_len 48 --pred_len 96 --d_model 512 --n_heads 8 --e_layers 2 --d_ff 2048 --patch_len 16 --stride 8 --llm_model_name "meta-llama/Llama-3-8b"
```

### Important Parameters

- `--model`: Set to "LLMPatchTST" to use this model
- `--data`: Set to "farm" to use the text dataset
- `--data_path`: Path to the text dataset file
- `--llm_model_name`: Name of the pre-trained LLM model to use (default: "meta-llama/Llama-3-8b")
- `--d_model`: Dimension of the model (used for both LLM projection and time series embedding)
- `--n_heads`: Number of attention heads
- `--e_layers`: Number of encoder layers
- `--patch_len`: Length of patches for PatchTST
- `--stride`: Stride for patch embedding

## Example

Here's a complete example of how to train and evaluate the model:

```bash
# Train the model
python experiment.py --is_training 1 --model LLMPatchTST --data farm --root_path ./dataset/ --data_path text_dataset_compact_in96_out96.csv --features M --seq_len 96 --label_len 48 --pred_len 96 --d_model 512 --n_heads 8 --e_layers 2 --d_ff 2048 --patch_len 16 --stride 8 --llm_model_name "meta-llama/Llama-3-8b" --train_epochs 10 --batch_size 32 --learning_rate 0.0001

# Test the model
python experiment.py --is_training 0 --model LLMPatchTST --data farm --root_path ./dataset/ --data_path text_dataset_compact_in96_out96.csv --features M --seq_len 96 --label_len 48 --pred_len 96 --d_model 512 --n_heads 8 --e_layers 2 --d_ff 2048 --patch_len 16 --stride 8 --llm_model_name "meta-llama/Llama-3-8b"
```

## Using Without LLM

If you don't have access to a pre-trained LLM, the model will automatically fall back to using random embeddings:

```bash
python experiment.py --is_training 1 --model LLMPatchTST --data farm --root_path ./dataset/ --data_path text_dataset_compact_in96_out96.csv --features M --seq_len 96 --label_len 48 --pred_len 96 --d_model 512 --n_heads 8 --e_layers 2 --d_ff 2048 --patch_len 16 --stride 8
```

## Implementation Details

### LLM Embedding Component

The LLM embedding component uses Hugging Face's `AutoModel` and `AutoTokenizer` to load a pre-trained LLM. The LLM parameters are frozen to avoid modifying the pre-trained weights. The LLM embeddings are projected to match the dimension of the time series features.

### Cross-Attention Mechanism

The cross-attention mechanism uses queries from the time series features and keys/values from the LLM embeddings. This allows the model to focus on relevant parts of the text description when making predictions.

### Handling Multiple Variables

For multivariate time series, each variable is processed separately with the cross-attention mechanism. This allows the model to focus on different parts of the text description for different variables.

## Notes

- The model works best with the `Dataset_TextPrompt` class, which provides text descriptions along with time series data.
- If using a different dataset class, the model will still work but will use random text embeddings.
- For best results, use a pre-trained LLM that has been fine-tuned on domain-specific text.