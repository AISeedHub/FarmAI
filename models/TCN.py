import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        # Đảm bảo kích thước khớp nhau bằng cách cắt tỉa nếu cần
        if out.size(2) != res.size(2):
            # Cắt bớt tensor lớn hơn để khớp với kích thước nhỏ hơn
            target_size = min(out.size(2), res.size(2))
            out = out[:, :, :target_size]
            res = res[:, :, :target_size]

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Encoder
        self.enc_embedding = nn.Linear(configs.enc_in, configs.d_model)
        num_channels = [configs.d_model] * configs.e_layers
        self.tcn = TemporalConvNet(configs.d_model, num_channels, kernel_size=3, dropout=configs.dropout)

        # Decoder
        self.dec_embedding = nn.Linear(configs.dec_in, configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # Chuyển đổi đầu ra
        self.output_proj = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Encoder
        enc_out = self.enc_embedding(x_enc)  # [Batch, seq_len, d_model]
        enc_out = enc_out.permute(0, 2, 1)  # [Batch, d_model, seq_len]

        tcn_output = self.tcn(enc_out)  # [Batch, d_model, seq_len]

        # Đảm bảo kích thước đầu ra của TCN đúng với kích thước cần thiết
        if tcn_output.size(2) != self.seq_len:
            # Trong trường hợp kích thước khác nhau, áp dụng nội suy tuyến tính
            tcn_output = F.interpolate(tcn_output, size=self.seq_len, mode='linear', align_corners=False)

        # Đảo lại permute
        tcn_output = tcn_output.permute(0, 2, 1)  # [Batch, seq_len, d_model]

        # Projection để có được dự đoán cuối cùng
        dec_out = self.projection(tcn_output)  # [Batch, seq_len, c_out]

        # Chiết xuất dự đoán cuối cùng
        dec_out = dec_out[:, -self.pred_len:, :]  # [Batch, pred_len, c_out]

        if self.output_attention:
            return dec_out, None
        else:
            return dec_out  # [Batch, pred_len, c_out]