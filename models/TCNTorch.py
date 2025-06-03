import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tcn import TCN as PyTorchTCN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        
        # Encoder
        self.enc_embedding = nn.Linear(configs.enc_in, configs.d_model)
        
        # Define dilations if not provided in configs
        dilations = getattr(configs, 'dilations', None)
        if dilations is None:
            # Default: exponentially increasing dilations
            num_layers = configs.e_layers
            dilations = [2 ** i for i in range(num_layers)]
        
        # Define number of channels
        num_channels = [configs.d_model] * configs.e_layers
        
        # Define kernel size
        kernel_size = getattr(configs, 'kernel_size', 4)
        
        # Define dilation reset
        dilation_reset = getattr(configs, 'dilation_reset', None)
        
        # Define dropout
        dropout = getattr(configs, 'dropout', 0.1)
        
        # Define causal
        causal = getattr(configs, 'causal', True)
        
        # Define use_norm
        use_norm = getattr(configs, 'use_norm', 'weight_norm')
        
        # Define activation
        activation = getattr(configs, 'activation', 'relu')
        
        # Define kernel_initializer
        kernel_initializer = getattr(configs, 'kernel_initializer', 'xavier_uniform')
        
        # Define use_skip_connections
        use_skip_connections = getattr(configs, 'use_skip_connections', False)
        
        # Define input_shape
        input_shape = getattr(configs, 'input_shape', 'NCL')
        
        # Define embedding_shapes
        embedding_shapes = getattr(configs, 'embedding_shapes', None)
        
        # Define embedding_mode
        embedding_mode = getattr(configs, 'embedding_mode', 'add')
        
        # Define use_gate
        use_gate = getattr(configs, 'use_gate', False)
        
        # Define lookahead
        lookahead = getattr(configs, 'lookahead', 0)
        
        # Define output_projection
        output_projection = getattr(configs, 'output_projection', None)
        
        # Define output_activation
        output_activation = getattr(configs, 'output_activation', None)
        
        # Create TCN from pytorch_tcn
        self.tcn = PyTorchTCN(
            num_inputs=configs.d_model,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dilation_reset=dilation_reset,
            dropout=dropout,
            causal=causal,
            use_norm=use_norm,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_skip_connections=use_skip_connections,
            input_shape=input_shape,
            embedding_shapes=embedding_shapes,
            embedding_mode=embedding_mode,
            use_gate=use_gate,
            lookahead=lookahead,
            output_projection=output_projection,
            output_activation=output_activation,
        )
        
        # Decoder
        self.dec_embedding = nn.Linear(configs.dec_in, configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Encoder
        enc_out = self.enc_embedding(x_enc)  # [Batch, seq_len, d_model]
        
        # TCN expects input shape [Batch, d_model, seq_len] for 'NCL' input_shape
        enc_out = enc_out.permute(0, 2, 1)  # [Batch, d_model, seq_len]
        
        # Apply TCN
        tcn_output = self.tcn(enc_out)  # [Batch, d_model, seq_len]
        
        # Ensure output has the correct shape
        if tcn_output.size(2) != self.seq_len:
            # In case the output size is different, apply linear interpolation
            tcn_output = F.interpolate(tcn_output, size=self.seq_len, mode='linear', align_corners=False)
        
        # Convert back to [Batch, seq_len, d_model]
        tcn_output = tcn_output.permute(0, 2, 1)  # [Batch, seq_len, d_model]
        
        # Project to output dimension
        dec_out = self.projection(tcn_output)  # [Batch, seq_len, c_out]
        
        # Extract prediction
        dec_out = dec_out[:, -self.pred_len:, :]  # [Batch, pred_len, c_out]
        
        if self.output_attention:
            return dec_out, None
        else:
            return dec_out  # [Batch, pred_len, c_out]