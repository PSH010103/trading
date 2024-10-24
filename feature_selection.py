import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
# import xgboost as xgb
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, GRU, Dense
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import pandas_ta as ta
# import gym
# from stable_baselines3 import DQN

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', ]


for ticker in tickers : 

    df = pd.read_csv(f'./preprocessed_data/{ticker}.csv')
    df_price = pd.read_csv(f'./stock_prices/{ticker}.csv')
    
    def find_nearest_future_date(date):
        future_dates = df_price[df_price['date'] > date]
        return future_dates['date'].iloc[0] if not future_dates.empty else pd.NaT

    # X = df[df['day_relative_to_event'] < df.groupby('event_id')['day_relative_to_event'].transform('max')].reset_index(drop=True)
    # y = df[df['day_relative_to_event'] > df.groupby('event_id')['day_relative_to_event'].transform('min')]['Close'].reset_index(drop=True)
    # y = (y/X['Close']-1) * 100
    X = df
    feature_cols = [col for col in X.columns if col not in ['date', 'target', 'event_id', 'day_relative_to_event', 'SMA_50']]
    X = X[feature_cols]
    tmp = df['date'].apply(find_nearest_future_date).copy()
    y = df_price[df_price['date'].isin(tmp)][['Close']].reset_index().drop(columns=['index'])

    # print(X, y)
    # feature_cols = [col for col in X.columns if col not in ['date', 'target', 'event_id', 'day_relative_to_event']]
    # X = X[feature_cols]


    # rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

    # # Initialize Boruta
    # feat_selector = BorutaPy(rf, n_estimators='auto', random_state=42)

    # # Fit Boruta
    # feat_selector.fit(X.values, y.values.ravel())

    # # Get selected features
    # selected_features = X.columns[feat_selector.support_].tolist()

    # # Subset the dataset
    # X_filtered = X[selected_features]

    X.to_csv(f'./processed_data/{ticker}_X.csv')
    y.to_csv(f'./processed_data/{ticker}_y.csv')

    # rf.fit(X, y)

    # # Get feature importances
    # importances = rf.feature_importances_
    # indices = np.argsort(importances)[::-1]

    # # Select top features
    # top_n = 20  # Number of top features to select
    # selected_features_rf = [X.columns[i] for i in indices[:top_n]]

    # # Subset the dataset
    # X_filtered_rf = X[selected_features_rf]

