import torch
import torch.nn as nn


class MeanPoolingModel(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim1=1024, hidden_dim2=512, hidden_dim3=256, hidden_dim4=128, dropout_rate=0.3):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.gelu4 = nn.GELU()
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(hidden_dim4, 1)
        # For residual connection: project out1 to hidden_dim4 if needed
        if hidden_dim1 != hidden_dim4:
            self.res_proj = nn.Linear(hidden_dim1, hidden_dim4)
        else:
            self.res_proj = nn.Identity()
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        pooled = torch.mean(x, dim=1)
        out = self.layernorm(pooled)
        out1 = self.fc1(out)
        out1 = self.gelu1(out1)
        out1 = self.dropout1(out1)
        out2 = self.fc2(out1)
        out2 = self.gelu2(out2)
        out2 = self.dropout2(out2)
        out3 = self.fc3(out2)
        out3 = self.gelu3(out3)
        out3 = self.dropout3(out3)
        out4 = self.fc4(out3)
        out4 = self.gelu4(out4)
        out4 = self.dropout4(out4)
        # Residual connection from out1 to out4
        out4 = out4 + self.res_proj(out1)
        out = self.fc5(out4)
        return out

class AttentionConvPoolingModel(nn.Module):
    def __init__(self, seq_len, input_dim, conv_dim=256, kernel_size=5, hidden_dim1=1024, hidden_dim2=512, hidden_dim3=256, hidden_dim4=128, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu_conv = nn.ReLU()
        self.attn = nn.Linear(conv_dim, 1)
        self.layernorm = nn.LayerNorm(conv_dim)
        self.fc1 = nn.Linear(conv_dim, hidden_dim1)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.gelu4 = nn.GELU()
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(hidden_dim4, 1)
        # For residual connection: project out1 to hidden_dim4 if needed
        if hidden_dim1 != hidden_dim4:
            self.res_proj = nn.Linear(hidden_dim1, hidden_dim4)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv1(x)      # (batch, conv_dim, seq_len)
        x = self.relu_conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, conv_dim)
        attn_weights = torch.softmax(self.attn(x), dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)
        out = self.layernorm(pooled)
        out1 = self.fc1(out)
        out1 = self.gelu1(out1)
        out1 = self.dropout1(out1)
        out2 = self.fc2(out1)
        out2 = self.gelu2(out2)
        out2 = self.dropout2(out2)
        out3 = self.fc3(out2)
        out3 = self.gelu3(out3)
        out3 = self.dropout3(out3)
        out4 = self.fc4(out3)
        out4 = self.gelu4(out4)
        out4 = self.dropout4(out4)
        # Residual connection from out1 to out4
        out4 = out4 + self.res_proj(out1)
        out = self.fc5(out4)
        return out