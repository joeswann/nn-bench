import torch
import torch.nn as nn
from torch.utils.data import Dataset
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data[idx], dtype=torch.float32),
            "target": torch.tensor(self.data[idx + 1, :20], dtype=torch.float32)  # Only predict stock returns
        }

class TimeSeriesData:
    def __init__(self):
        self.data, self.input_size, self.output_size = self.prepare_data()
        self.dataset = TimeSeriesDataset(self.data)

    def get_top_sp500(self, n=20):
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return sp500.nlargest(n, 'Market Cap')['Symbol'].tolist()

    def get_stock_data(self, symbols):
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=1)
        data = yf.download(symbols, start=start_date, end=end_date)
        return data['Adj Close']

    def get_sentiment_data(self, symbols):
        sia = SentimentIntensityAnalyzer()
        sentiment_data = {}
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            daily_sentiment = []
            
            for item in news:
                sentiment = sia.polarity_scores(item['title'])
                daily_sentiment.append(sentiment['compound'])
            
            if daily_sentiment:
                sentiment_data[symbol] = np.mean(daily_sentiment)
            else:
                sentiment_data[symbol] = 0
        
        return pd.Series(sentiment_data)

    def prepare_data(self):
        symbols = self.get_top_sp500()
        stock_data = self.get_stock_data(symbols)
        sentiment_data = self.get_sentiment_data(symbols)
        
        # Combine stock data and sentiment data
        combined_data = stock_data.copy()
        for symbol in symbols:
            combined_data[f'{symbol}_sentiment'] = sentiment_data[symbol]
        
        # Fill missing values and calculate returns
        combined_data = combined_data.ffill().bfill()
        returns = combined_data.pct_change().dropna()
        
        # Normalize the data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(returns)
        
        input_size = normalized_data.shape[1]
        output_size = len(symbols)  # We're only predicting stock returns, not sentiment
        
        return normalized_data, input_size, output_size

    def get_data(self):
        return self.dataset, self.input_size, self.output_size

    def get_criterion(self):
        return nn.MSELoss()

    def get_hyperparams(self):
        return {
            'steps': 10,
            'step_size': 0.01,
            'solver': 'RungeKutta',
            'adaptive': True
        }
