# PowerShell script to run TCN model for time series forecasting
# Usage: .\run_tcn.ps1 [OPTIONS]
# Options:
#   -data: Dataset to use (default: custom)
#   -features: Forecasting task (M: multivariate predict multivariate, S: univariate predict univariate, MS: multivariate predict univariate)
#   -seq_len: Input sequence length
#   -pred_len: Prediction sequence length
#   -enc_in: Number of input variables
#   -c_out: Number of output variables
#   -target: Target variable to predict (default: CO2)

param (
    [string]$data = "custom",
    [string]$features = "M",
    [int]$seq_len = 96,
    [int]$pred_len = 96,
    [int]$enc_in = 9,
    [int]$c_out = 9,
    [string]$target = "CO2"
)

# Display the configuration
Write-Host "Running TCN model with the following configuration:"
Write-Host "Data: $data"
Write-Host "Features: $features"
Write-Host "Sequence Length: $seq_len"
Write-Host "Prediction Length: $pred_len"
Write-Host "Input Variables: $enc_in"
Write-Host "Output Variables: $c_out"
Write-Host "Target Variable: $target"

# Run the TCN model
python experiment.py `
  --is_training 1 `
  --model TCN `
  --data $data `
  --root_path .\dataset\ `
  --data_path data-com-latest-modified.csv `
  --features $features `
  --seq_len $seq_len `
  --pred_len $pred_len `
  --enc_in $enc_in `
  --dec_in $enc_in `
  --c_out $c_out `
  --target $target `
  --d_model 512 `
  --e_layers 2 `
  --d_ff 2048 `
  --dropout 0.05 `
  --batch_size 32 `
  --learning_rate 0.0001 `
  --train_epochs 10

Write-Host "TCN model training completed!"
