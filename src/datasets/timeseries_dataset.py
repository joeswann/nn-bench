import torch
import torch.nn as nn
from torch.utils.data import Dataset
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml
import logging

# Load configuration
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        input_data = self.data[idx:idx+self.sequence_length]
        target_data = self.data[idx+1:idx+self.sequence_length+1]
        return {
            "input": torch.tensor(input_data, dtype=torch.float32),
            "target": torch.tensor(target_data, dtype=torch.float32)
        }

class TimeSeriesData:
    def __init__(self, config):
        self.config = config
        self.symbols = config.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'])
        self.sequence_length = config['sequence_length']
        self.data, self.input_size, self.output_size = self.prepare_data()
        self.dataset = TimeSeriesDataset(self.data, self.sequence_length)

    def get_stock_data(self):
        all_data = []
        for symbol in self.symbols:
            try:
                data = yf.download(symbol, period=self.config['period'], interval=self.config['interval'])
                if data.empty:
                    logging.warning(f"No data found for {symbol}. Skipping this symbol.")
                    continue
                all_data.append(data)
            except Exception as e:
                logging.error(f"Error downloading data for {symbol}: {str(e)}")
        
        if not all_data:
            raise ValueError("No valid data found for any symbol.")
        
        return pd.concat(all_data, axis=1, keys=[d.columns.name for d in all_data])

    def prepare_data(self):
        stock_data = self.get_stock_data()
        
        # Use all available columns for all symbols
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        final_data = stock_data.loc[:, (slice(None), features)]
        
        # Handle NaN values
        final_data = final_data.ffill().bfill()
        
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(final_data)
        
        input_size = normalized_data.shape[1]
        output_size = len(stock_data.columns.levels[0])  # Number of valid symbols

        return normalized_data, input_size, output_size

    def get_data(self):
        return self.dataset, self.input_size, self.output_size

    def get_criterion(self):
        return nn.MSELoss()

    def get_hyperparams(self):
        return {
            'steps': config['ltc']['steps'],
            'step_size': config['ltc']['step_size'],
            'solver': config['ltc']['solver'],
            'adaptive': config['ltc']['adaptive']
        }
