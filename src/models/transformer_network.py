import torch
import torch.nn as nn
import math

class TransformerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout=0.1):
        super(TransformerNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.hidden_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_layer(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
