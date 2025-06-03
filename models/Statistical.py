import torch
import torch.nn as nn
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

class Model(nn.Module):
    """
    Statistical model for time series forecasting.
    Implements several simple statistical methods:
    - naive_last: uses the last value of the input sequence for all future predictions
    - naive_mean: uses the mean of the input sequence for all future predictions
    - naive_seasonal: uses the last season's values (assumes seasonality equals pred_len)
    - drift: linear extrapolation based on the first and last points of the input
    - arima: uses ARIMA (AutoRegressive Integrated Moving Average) model for forecasting
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.method = getattr(configs, 'stat_method', 'naive_last')
        self.output_attention = configs.output_attention

        # ARIMA parameters (p, d, q)
        self.arima_order = getattr(configs, 'arima_order', (1, 1, 0))

        # No trainable parameters for statistical models
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc: [Batch, seq_len, enc_in]
        # We'll use the last dimension for forecasting

        batch_size, _, enc_in = x_enc.shape

        # Initialize prediction tensor
        pred = torch.zeros((batch_size, self.pred_len, enc_in), device=x_enc.device)

        if self.method == 'naive_last':
            # Use the last value for all future predictions
            last_value = x_enc[:, -1:, :]  # Shape: [Batch, 1, enc_in]
            pred = last_value.repeat(1, self.pred_len, 1)

        elif self.method == 'naive_mean':
            # Use the mean of the input sequence for all future predictions
            mean_value = torch.mean(x_enc, dim=1, keepdim=True)  # Shape: [Batch, 1, enc_in]
            pred = mean_value.repeat(1, self.pred_len, 1)

        elif self.method == 'naive_seasonal':
            # Use the last season's values (assumes seasonality equals pred_len)
            # If pred_len > seq_len, we'll repeat the pattern
            for i in range(self.pred_len):
                idx = -(self.pred_len - i % self.pred_len)
                if abs(idx) <= self.seq_len:
                    pred[:, i, :] = x_enc[:, idx, :]
                else:
                    # If we don't have enough history, use the last value
                    pred[:, i, :] = x_enc[:, -1, :]

        elif self.method == 'drift':
            # Linear extrapolation based on the first and last points
            first_value = x_enc[:, 0, :]  # Shape: [Batch, enc_in]
            last_value = x_enc[:, -1, :]  # Shape: [Batch, enc_in]

            # Calculate slope
            slope = (last_value - first_value) / (self.seq_len - 1)  # Shape: [Batch, enc_in]

            # Generate predictions
            for i in range(self.pred_len):
                pred[:, i, :] = last_value + slope * (i + 1)

        elif self.method == 'arima':
            # ARIMA forecasting for each time series in the batch
            x_enc_np = x_enc.detach().cpu().numpy()

            # Process each batch and feature separately
            for b in range(batch_size):
                for f in range(enc_in):
                    # Get the time series for this batch and feature
                    history = x_enc_np[b, :, f]

                    try:
                        # Fit ARIMA model
                        model = ARIMA(history, order=self.arima_order)
                        model_fit = model.fit()

                        # Generate forecast
                        forecast = model_fit.forecast(steps=self.pred_len)

                        # Convert forecast to tensor and store in pred
                        pred[b, :, f] = torch.tensor(forecast, device=x_enc.device)
                    except Exception as e:
                        # If ARIMA fails, fall back to drift method
                        print(f"ARIMA failed for batch {b}, feature {f}: {e}")
                        print("Falling back to drift method")

                        # Simple drift method as fallback
                        first_val = history[0]
                        last_val = history[-1]
                        slope = (last_val - first_val) / (self.seq_len - 1)

                        for i in range(self.pred_len):
                            pred[b, i, f] = last_val + slope * (i + 1)

        else:
            # Default to naive_last if method is not recognized
            last_value = x_enc[:, -1:, :]
            pred = last_value.repeat(1, self.pred_len, 1)

        if self.output_attention:
            return pred, None
        else:
            return pred  # [B, L, D]
