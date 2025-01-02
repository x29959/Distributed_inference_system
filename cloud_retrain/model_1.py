import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, num_classes, output_size, units, dropout=0.5):
        super(CNN_LSTM, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.dropout_conv = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=units,
            num_layers=12,
            dropout=dropout,
            batch_first=True, 
            bidirectional=False
        )

        self.Relu = nn.ReLU()
        self.dropout_relu = nn.Dropout(p=dropout)

        self.Linear_1 = nn.Linear(
            in_features=units,
            out_features=num_classes
        )
        self.dropout_linear = nn.Dropout(p=dropout)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.tensor(x).permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.dropout_conv(x)
        x = self.Relu(x)
        x = self.dropout_relu(x)
        x = torch.tensor(x).permute(0, 2, 1)  # torch.Size([64,28,64])
        h_n, c_n = self.lstm(x)
        x = self.Linear_1(h_n[:, -1, :])
        x = self.dropout_linear(x)
        # x = self.softmax(x)
        return x


class CBiLSTM(nn.Module):
    def __init__(self, input_size, output_size, units, dropout=0.5):
        super(CBiLSTM, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.dropout_conv = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=units,
            num_layers=10,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        self.relu = nn.ReLU()
        self.dropout_relu = nn.Dropout(p=dropout)

        self.linear_1 = nn.Linear(
            in_features=units * 2,
            out_features=output_size
        )
        self.dropout_linear = nn.Dropout(p=dropout)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.tensor(x).permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.dropout_conv(x)
        x = self.relu(x)
        x = self.dropout_relu(x)
        x = torch.tensor(x).permute(0, 2, 1)
        h_n, c_n = self.lstm(x)
        x = self.linear_1(h_n[:, -1, :])
        x = self.dropout_linear(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer_rev_Complex(nn.Module):
    def __init__(self, input_size, num_classes, units, heads, dropout=0.3):
        super(Transformer_rev_Complex, self).__init__()
        self.conv1d_1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=units, 
            kernel_size=3, 
            padding=1
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels=units, 
            out_channels=units, 
            kernel_size=3, 
            padding=1
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoding = PositionalEncoding(d_model=units, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=units, 
            nhead=heads, 
            dim_feedforward=units,
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.linear = nn.Linear(in_features=units, out_features=num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # permute to (batch, features, seq_len)
        x = self.conv1d_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1d_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # back to (batch, seq_len, features)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        x = self.linear(x[:, -1, :])  # take features from the last time step
        return torch.softmax(x, dim=1)  # remain output sum to 1
        # return F.log_softmax(x, dim=1)


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(GRUClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # GRU層
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # 一個全連接層，用於從GRU的輸出中產生最終的分類結果
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隱藏狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向傳播GRU
        # out: [batch_size, seq_length, hidden_size]
        out, _ = self.gru(x, h0)

        # 取序列中的最後一個時間點的輸出
        out = out[:, -1, :]

        # Apply dropout
        out = self.dropout(out)

        # 經過全連接層得到最終的分類結果
        out = self.fc(out)
        return out


class CRNN(nn.Module):
    def __init__(self, input_size, output_size, units, dropout=0.5):
        super(CRNN, self).__init__()
        # 卷積層配置，保持與前面的模型相同
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.dropout_conv = nn.Dropout(p=dropout)

        # 使用基本的 RNN 單元
        self.rnn = nn.RNN(
            input_size=output_size,
            hidden_size=units,
            num_layers=8,
            dropout=dropout if 8 > 1 else 0.0,
            batch_first=True,
            nonlinearity='relu'
        )
        self.dropout_rnn = nn.Dropout(p=dropout)

        self.relu = nn.ReLU()
        self.dropout_relu = nn.Dropout(p=dropout)

        self.linear_1 = nn.Linear(
            in_features=units,  # 單向 RNN 的輸出特徵數量
            out_features=output_size
        )
        self.dropout_linear = nn.Dropout(p=dropout)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 轉置以適配卷積層期望的輸入維度
        x = self.conv1d(x)
        x = self.dropout_conv(x)
        x = self.relu(x)
        x = self.dropout_relu(x)
        x = x.permute(0, 2, 1)  # 再次轉置以適配 RNN 的輸入維度
        output, h_n = self.rnn(x)  # RNN 的輸出
        output = self.dropout_rnn(output)
        x = self.linear_1(output[:, -1, :])  # 使用最後一個時刻的輸出
        x = self.dropout_linear(x)
        # x = self.softmax(x)
        return x