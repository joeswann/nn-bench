import torch
import torch.nn as nn
from torch.utils.data import Dataset
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, num_stocks):
        self.data = data
        self.sequence_length = sequence_length
        self.num_stocks = num_stocks

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        input_data = self.data[idx:idx+self.sequence_length]
        target_data = self.data[idx+1:idx+self.sequence_length+1, :self.num_stocks]
        return {
            "input": torch.tensor(input_data, dtype=torch.float32),
            "target": torch.tensor(target_data, dtype=torch.float32)
        }

class TimeSeriesData:
    def __init__(self, sequence_length=32):
        self.sequence_length = sequence_length
        self.data, self.input_size, self.output_size = self.prepare_data()
        self.dataset = TimeSeriesDataset(self.data, self.sequence_length, self.output_size)

    def get_top_sp500(self, n=40):
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        if 'Market Cap' in sp500.columns:
            return sp500.nlargest(n, 'Market Cap')['Symbol'].tolist()
        else:
            return sp500.head(n)['Symbol'].tolist()

    def get_stock_data(self, symbols):
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=2)  # Extend to 2 years for more data
        data = yf.download(symbols, start=start_date, end=end_date)
        return data['Adj Close'].ffill()  # Forward fill missing values

    def get_sentiment_data(self, symbols, date_range):
        sia = SentimentIntensityAnalyzer()
        sentiment_data = pd.DataFrame(index=date_range)
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            daily_sentiment = pd.Series(index=date_range, name=f'{symbol}_sentiment')
            
            for item in news:
                news_date = pd.Timestamp(item['providerPublishTime'], unit='s')
                if news_date in date_range:
                    sentiment = sia.polarity_scores(item['title'])
                    daily_sentiment[news_date] = sentiment['compound']
            
            # Use 7-day moving average to smooth sentiment data
            daily_sentiment = daily_sentiment.rolling(window=7, min_periods=1).mean()
            daily_sentiment = daily_sentiment.ffill().fillna(0)
            sentiment_data = pd.concat([sentiment_data, daily_sentiment], axis=1)
        
        return sentiment_data

    def prepare_data(self):
        symbols = self.get_top_sp500()
        stock_data = self.get_stock_data(symbols)
        
        # Calculate returns for stock data (only for trading days)
        returns = stock_data.pct_change().dropna()
        
        # Get sentiment data for the same date range as the stock data
        sentiment_data = self.get_sentiment_data(symbols, returns.index)
        
        # Combine returns and sentiment data
        final_data = pd.concat([returns, sentiment_data], axis=1)
        final_data = final_data.dropna()
        
        print("Final data shape:", final_data.shape)
        print("Final data head:\n", final_data.head())
        print("Final data description:\n", final_data.describe())
        
        # Normalize the data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(final_data)
        
        input_size = normalized_data.shape[1]
        output_size = len(symbols)  # Number of stocks (returns to predict)

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
