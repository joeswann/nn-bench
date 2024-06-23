import torch
import torch.nn as nn
import logging
import yaml

with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

logging.basicConfig(level=logging.INFO, format=config['logging']['format'])

class CNNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, kernel_size=3):
        super(CNNNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.shape_logging = config['logging']['shape_logging']

        self.convs = nn.ModuleList([
            nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size, padding=(kernel_size - 1) // 2)
            for i in range(num_layers)
        ])
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv1d(hidden_size, input_size, 1)  # 1x1 convolution to adjust the number of channels to match input

    def forward(self, x):
        # Check input shape and reshape if necessary
        if len(x.shape) == 3:
            batch_size, seq_len, num_features = x.shape
            x = x.transpose(1, 2)  # (batch_size, num_features, seq_len)
        elif len(x.shape) == 2:
            batch_size, seq_len = x.shape
            x = x.unsqueeze(1)  # (batch_size, 1, seq_len)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        if self.shape_logging:
            logging.info(f"Input shape after reshaping: {x.shape}")
        
        # Apply convolutions
        for i, conv in enumerate(self.convs):
            x = self.relu(conv(x))
            if self.shape_logging:
                logging.info(f"Shape after conv {i+1}: {x.shape}")
        
        # Apply final convolution to adjust the number of channels
        x = self.final_conv(x)
        if self.shape_logging:
            logging.info(f"Shape after final conv: {x.shape}")
        
        # Transpose the output to match the expected shape (batch_size, seq_len, input_size)
        output = x.transpose(1, 2)
        
        if self.shape_logging:
            logging.info(f"Final output shape: {output.shape}")
        
        return output
