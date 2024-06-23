import torch
import torch.nn as nn

class CNNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, kernel_size=3):
        super(CNNNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(num_layers)
        ])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (batch, hidden_size, seq_len)
        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_size)
        output = self.output_layer(x)
        return output
