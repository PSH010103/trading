import pandas as pd
from datetime import datetime
import numpy as np
from scipy import stats
import pandas_ta as ta
import matplotlib.pyplot as plt

# S&P 500 티커 목록 (예시로 일부만 사용)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
ticker = tickers[0]

leading_indicators = pd.read_csv('./leadingindicators/fred.csv', parse_dates=['date'])
leading_indicators.sort_values('date', inplace=True)
leading_indicators.reset_index(drop=True, inplace=True)


stock_data  = pd.read_csv(f'./stock_prices/{ticker}.csv', parse_dates=['date'])
stock_data .sort_values('date', inplace=True)
stock_data .reset_index(drop=True, inplace=True)


earning_data = pd.read_csv(f'./earnings/{ticker}.csv', parse_dates=['date'])
earning_data.sort_values('date', inplace=True)
earning_data.reset_index(drop=True, inplace=True)
earning_data['EPS_diff'] = pd.to_numeric(earning_data['Actual EPS']) - pd.to_numeric(earning_data['Estimated EPS'])
earning_data = earning_data[['date', 'EPS_diff']]

earning_dates =  earning_data['date'].unique()

df = pd.merge_asof(
            stock_data,
            earning_data,
            left_on='date',
            right_on='date',
            by=None,
            direction='backward')

df = pd.merge_asof(
            df,
            leading_indicators,
            left_on='date',
            right_on='date',
            by=None,
            direction='backward')

print(df)


N = 5 # You can adjust this as needed

# Initialize a list to store event window DataFrames
event_windows = []

for idx, event in earning_data.iterrows():
    event_date = event['date']
    
    try:
        event_idx = df.index[df['date'] == event_date][0]
    except IndexError:
        print(f"Aligned date {event_date} not found in other_data. Skipping.")
        continue

    start_idx = max(event_idx - N, 0)
    end_idx = min(event_idx + N + 1, len(df))  # +1 because slice is exclusive
    
    # Extract window
    window = df.iloc[start_idx:end_idx].copy()
    
    # Add event-specific information
    window['event_id'] = idx  # Unique identifier for the event
    window['day_relative_to_event'] = window.index - event_idx  # Relative day (e.g., -10 to +10)
    
    # Merge earning data into the window
    # If you want to include earning data as features, you can add the earning metrics to each row in the window
    earning_columns = earning_data.columns.tolist()
    earning_columns.remove('date')  # Exclude original 'date' to avoid duplication
    
    for col in earning_columns:
        window[col] = event[col]
    
    # Append to the list
    event_windows.append(window)
    
    
processed_data = pd.concat(event_windows, ignore_index=True)

print("\nProcessed Data Sample:")
print(processed_data.head())
print("\nProcessed Data Shape:", processed_data.shape)


# Handle missing values
# Forward fill, then backward fill as a fallback
processed_data.fillna(method='ffill', inplace=True)
processed_data.fillna(method='bfill', inplace=True)

# Remove any remaining rows with NaN values
processed_data.dropna(inplace=True)

# Outlier removal using Z-score
numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()

# Exclude 'day_relative_to_event' and 'event_id' from outlier detection
exclude_cols = ['day_relative_to_event', 'event_id']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

# Apply Z-score and filter out outliers beyond 3 standard deviations
processed_data = processed_data[(np.abs(stats.zscore(processed_data[numeric_cols])) < 3).all(axis=1)]

print("\nProcessed Data Shape after Cleaning:", processed_data.shape)
processed_data['SMA_20'] = processed_data['Close'].rolling(window=20).mean()
processed_data['SMA_50'] = processed_data['Close'].rolling(window=50).mean()
processed_data['RSI'] = ta.rsi(processed_data['Close'], length=14)
macd = ta.macd(processed_data['Close'])
processed_data['MACD'] = macd['MACDh_12_26_9']
processed_data = processed_data.set_index('date')
processed_data.fillna(method='ffill', inplace=True)
processed_data.fillna(method='bfill', inplace=True)
processed_data.to_csv(f'./processed_data/{ticker}.csv')

