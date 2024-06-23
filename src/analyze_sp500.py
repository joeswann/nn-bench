import yfinance as yf
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import yaml
import os

def load_config(config_path='config.yml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_stock_data(ticker, period="5y"):
    """Fetch stock data for a given ticker"""
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

def get_news_sentiment(ticker):
    """Get news sentiment for a ticker"""
    stock = yf.Ticker(ticker)
    news = stock.news
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(article['title'])['compound'] for article in news]
    return pd.Series(sentiments).mean()

def analyze_sp500():
    config = load_config()
    symbols = config['dataset']['timeseries']['symbols']
    period = config['dataset']['timeseries']['period']

    all_data = []
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        stock_data = get_stock_data(symbol, period)
        sentiment = get_news_sentiment(symbol)
        
        # Add sentiment to the stock data
        stock_data['Sentiment'] = sentiment
        stock_data['Symbol'] = symbol
        
        all_data.append(stock_data)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, axis=0, sort=False)

    # Reset index to make 'Symbol' the first column
    combined_data = combined_data.reset_index()
    
    # Reorder columns to make 'Symbol' the first column
    columns = ['Symbol'] + [col for col in combined_data.columns if col != 'Symbol']
    combined_data = combined_data[columns]

    # Save the combined data to a single CSV file in the root folder
    filename = 'sp500_data.csv'
    combined_data.to_csv(filename, index=False)
    print(f"All data saved to {filename}")

    print("All data fetched, analyzed, and saved successfully.")

if __name__ == "__main__":
    analyze_sp500()
