from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from models import * 

# Split the data
X_train = pd.read_csv(f'./train_X.csv', index_col=0)
y_train = pd.read_csv(f'./train_y.csv', index_col=0)

X_val = pd.read_csv(f'./valid_X.csv', index_col=0)
y_val = pd.read_csv(f'./valid_y.csv', index_col=0)

X_test = pd.read_csv(f'./test_X.csv', index_col=0)
y_test = pd.read_csv(f'./test_y.csv', index_col=0)



# Initialize scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit scalers on training data
X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()


# Transform validation and test data
X_val_scaled = pd.DataFrame(scaler_X.transform(X_val), index=X_val.index, columns=X_val.columns)
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

X_test_scaled = pd.DataFrame(scaler_X.transform(X_test), index=X_test.index, columns=X_test.columns)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()



# Initialize and train the model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_scaled)

# Validation predictions
y_val_pred_scaled = lr_model.predict(X_val_scaled)
val_mape = mean_absolute_percentage_error(y_val_scaled, y_val_pred_scaled)
print(f'Linear Regression Validation MAPE: {val_mape:.4f}')

# Test predictions
y_test_pred_scaled = lr_model.predict(X_test_scaled)
test_mape = mean_absolute_percentage_error(y_test_scaled, y_test_pred_scaled)
print(f'Linear Regression Test MAPE: {test_mape:.4f}')

# Inverse transform predictions
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = y_test.values.flatten()

# Calculate MAPE on original scale
test_mape_original_lr = mean_absolute_percentage_error(y_test_actual, y_test_pred)
print(f'Linear Regression Test MAPE (Original Scale): {test_mape_original_lr:.4f}')



# Define parameter grid
param_grid_dt = {
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10],
}

# Initialize model
dt_model = DecisionTreeRegressor(random_state=42)

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Grid Search
grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
grid_search_dt.fit(X_train_scaled, y_train_scaled)

# Best parameters
print(f'Best Parameters for Decision Tree: {grid_search_dt.best_params_}')

# Validation performance
y_val_pred_scaled = grid_search_dt.predict(X_val_scaled)
val_mape = mean_absolute_percentage_error(y_val_scaled, y_val_pred_scaled)
print(f'Decision Tree Validation MAPE: {val_mape:.4f}')

# Test performance
y_test_pred_scaled = grid_search_dt.predict(X_test_scaled)
test_mape = mean_absolute_percentage_error(y_test_scaled, y_test_pred_scaled)
print(f'Decision Tree Test MAPE: {test_mape:.4f}')

# Inverse transform predictions
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = y_test.values.flatten()

# Calculate MAPE on original scale
test_mape_original_dt = mean_absolute_percentage_error(y_test_actual, y_test_pred)
print(f'Decision Tree Test MAPE (Original Scale): {test_mape_original_dt:.4f}')



# Define parameter grid
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1],
}

# Initialize model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Grid Search
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
grid_search_xgb.fit(X_train_scaled, y_train_scaled)

# Best parameters
print(f'Best Parameters for XGBoost: {grid_search_xgb.best_params_}')

# Validation performance
y_val_pred_scaled = grid_search_xgb.predict(X_val_scaled)
val_mape = mean_absolute_percentage_error(y_val_scaled, y_val_pred_scaled)
print(f'XGBoost Validation MAPE: {val_mape:.4f}')

# Test performance
y_test_pred_scaled = grid_search_xgb.predict(X_test_scaled)
test_mape = mean_absolute_percentage_error(y_test_scaled, y_test_pred_scaled)
print(f'XGBoost Test MAPE: {test_mape:.4f}')

# Inverse transform predictions
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = y_test.values.flatten()

# Calculate MAPE on original scale
test_mape_original_xgb = mean_absolute_percentage_error(y_test_actual, y_test_pred)
print(f'XGBoost Test MAPE (Original Scale): {test_mape_original_xgb:.4f}')



sequence_length = 10

# Training set
train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, sequence_length)

# Validation set
val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, sequence_length)

# Test set
test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, sequence_length)


hidden_sizes = [32, 64]
num_layers_list = [1, 2]
learning_rates = [0.001, 0.0001]
batch_sizes = [32, 64]



input_size = X_train_scaled.shape[1]

best_mape_lstm = float('inf')
best_params_lstm = {}
best_model_lstm = None

for hidden_size in hidden_sizes:
    for num_layers in num_layers_list:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f'Training LSTM with hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}, batch_size={batch_size}')
                
                # Update DataLoaders with new batch_size
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Initialize model
                model_lstm = LSTMModel(input_size, hidden_size=hidden_size, num_layers=num_layers)
                
                # Train model
                model_lstm, val_mape = train_model(model_lstm, train_loader, val_loader, num_epochs=1, learning_rate=lr)
                
                # Update best model if validation MAPE improves
                if val_mape < best_mape_lstm:
                    best_mape_lstm = val_mape
                    best_params_lstm = {
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'learning_rate': lr,
                        'batch_size': batch_size
                    }
                    best_model_lstm = model_lstm

print(f'Best Parameters for LSTM: {best_params_lstm}')
print(f'Best Validation MAPE for LSTM: {best_mape_lstm:.4f}')

# Update test_loader with best batch size
test_loader = DataLoader(test_dataset, batch_size=best_params_lstm['batch_size'], shuffle=False)

# Evaluate on Test Set
best_model_lstm.eval()
test_predictions = []
test_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = best_model_lstm(X_batch)
        test_predictions.extend(outputs.squeeze().numpy())
        test_targets.extend(y_batch.numpy())

# Inverse transform predictions and targets
test_predictions_inv = scaler_y.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
test_targets_inv = scaler_y.inverse_transform(np.array(test_targets).reshape(-1, 1)).flatten()

# Calculate MAPE
test_mape_lstm = mean_absolute_percentage_error(test_targets_inv, test_predictions_inv)
print(f'LSTM Test MAPE: {test_mape_lstm:.4f}')



best_mape_gru = float('inf')
best_params_gru = {}
best_model_gru = None

for hidden_size in hidden_sizes:
    for num_layers in num_layers_list:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f'Training GRU with hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}, batch_size={batch_size}')
                
                # Update DataLoaders with new batch_size
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Initialize model
                model_gru = GRUModel(input_size, hidden_size=hidden_size, num_layers=num_layers)
                
                # Train model
                model_gru, val_mape = train_model(model_gru, train_loader, val_loader, num_epochs=1, learning_rate=lr)
                
                # Update best model if validation MAPE improves
                if val_mape < best_mape_gru:
                    best_mape_gru = val_mape
                    best_params_gru = {
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'learning_rate': lr,
                        'batch_size': batch_size
                    }
                    best_model_gru = model_gru

print(f'Best Parameters for GRU: {best_params_gru}')
print(f'Best Validation MAPE for GRU: {best_mape_gru:.4f}')




# Update test_loader with best batch size
test_loader = DataLoader(test_dataset, batch_size=best_params_gru['batch_size'], shuffle=False)

# Evaluate on Test Set
best_model_gru.eval()
test_predictions = []
test_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = best_model_gru(X_batch)
        test_predictions.extend(outputs.squeeze().numpy())
        test_targets.extend(y_batch.numpy())

# Inverse transform predictions and targets
test_predictions_inv = scaler_y.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
test_targets_inv = scaler_y.inverse_transform(np.array(test_targets).reshape(-1, 1)).flatten()

# Calculate MAPE
test_mape_gru = mean_absolute_percentage_error(test_targets_inv, test_predictions_inv)
print(f'GRU Test MAPE: {test_mape_gru:.4f}')



to_save = ""
print("\nModel Performance Comparison:")
print(f"Linear Regression Test MAPE (Original Scale): {test_mape_original_lr:.4f}")
print(f"Decision Tree Test MAPE (Original Scale): {test_mape_original_dt:.4f}")
print(f"XGBoost Test MAPE (Original Scale): {test_mape_original_xgb:.4f}")
print(f"LSTM Test MAPE: {test_mape_lstm:.4f}")
print(f"GRU Test MAPE: {test_mape_gru:.4f}")

to_save += (f"\nModel Performance Comparison:\n")
to_save +=(f"Linear Regression Test MAPE (Original Scale): {test_mape_original_lr:.4f}\n")
to_save +=(f"Decision Tree Test MAPE (Original Scale): {test_mape_original_dt:.4f}\n")
to_save +=(f"XGBoost Test MAPE (Original Scale): {test_mape_original_xgb:.4f}\n")
to_save +=(f"LSTM Test MAPE: {test_mape_lstm:.4f}\n")
to_save +=(f"GRU Test MAPE: {test_mape_gru:.4f}\n")
with open('./perf.txt', 'a') as f:
    f.write(to_save)


best_model = grid_search_xgb
X = pd.concat([X_train], ignore_index=True)
y = pd.concat([y_train], ignore_index=True)
best_model.fit(X_train, y_train)
y_test_pred_scaled = best_model.predict(X_test_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = y_test.values.flatten()

# Generate signals
signals = np.where(y_test_pred > X_test['Close'], 1, -1)  # 1 for Long, -1 for Short

# Calculate daily returns
returns = signals * (y_test_actual - X_test['Close']) / X_test['Close']

# Cumulative returns
cumulative_returns = np.cumprod(1 + returns) - 1



# Calculate Sharpe Ratio
mean_return = np.mean(returns)
std_return = np.std(returns)
sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized Sharpe Ratio

# Calculate Maximum Drawdown
cum_returns = np.cumprod(1 + returns)
roll_max = np.maximum.accumulate(cum_returns)
drawdown = (cum_returns - roll_max) / roll_max
max_drawdown = drawdown.min()

print(f'Total Test Return: {cumulative_returns.iloc[-1]}')
print(f'Sharpe Ratio on Test Set: {sharpe_ratio:.4f}')
print(f'Maximum Drawdown on Test Set: {max_drawdown:.4f}')

