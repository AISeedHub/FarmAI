#!/bin/bash

# Script to run TCN model for time series forecasting
# Usage: ./run_tcn.sh [OPTIONS]
# Options:
#   --data: Dataset to use (default: custom)
#   --features: Forecasting task (M: multivariate predict multivariate, S: univariate predict univariate, MS: multivariate predict univariate)
#   --seq_len: Input sequence length
#   --pred_len: Prediction sequence length
#   --enc_in: Number of input variables
#   --c_out: Number of output variables
#   --target: Target variable to predict (default: CO2)

# Default values
DATA="custom"
FEATURES="M"
SEQ_LEN=96
PRED_LEN=96
ENC_IN=9
C_OUT=9
TARGET="CO2"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)
      DATA="$2"
      shift 2
      ;;
    --features)
      FEATURES="$2"
      shift 2
      ;;
    --seq_len)
      SEQ_LEN="$2"
      shift 2
      ;;
    --pred_len)
      PRED_LEN="$2"
      shift 2
      ;;
    --enc_in)
      ENC_IN="$2"
      shift 2
      ;;
    --c_out)
      C_OUT="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Display the configuration
echo "Running TCN model with the following configuration:"
echo "Data: $DATA"
echo "Features: $FEATURES"
echo "Sequence Length: $SEQ_LEN"
echo "Prediction Length: $PRED_LEN"
echo "Input Variables: $ENC_IN"
echo "Output Variables: $C_OUT"
echo "Target Variable: $TARGET"

# Run the TCN model
python experiment.py \
  --is_training 1 \
  --model TCN \
  --data $DATA \
  --root_path ./dataset/ \
  --data_path data-com-latest-modified.csv \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $C_OUT \
  --target $TARGET \
  --d_model 512 \
  --e_layers 2 \
  --d_ff 2048 \
  --dropout 0.05 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10

echo "TCN model training completed!"
