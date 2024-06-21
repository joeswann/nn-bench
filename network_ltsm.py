import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, use_embedding=True):
        super(LSTMNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_embedding = use_embedding

        if use_embedding:
            self.embedding = nn.Embedding(input_size, hidden_size)
        else:
            self.input_layer = nn.Linear(input_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        if self.use_embedding:
            input_sequence = self.embedding(input_sequence)
        else:
            input_sequence = self.input_layer(input_sequence)

        output, _ = self.lstm(input_sequence)
        output = self.output_layer(output)
        return output
