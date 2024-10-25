import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=10):
        self.X = X.values
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.sequence_length]
        y_seq = self.y[idx+self.sequence_length]
        return torch.Tensor(X_seq), torch.Tensor([y_seq])
    

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_mape = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                
                # Collect predictions and targets for MAPE
                val_predictions.extend(outputs.squeeze().numpy())
                val_targets.extend(y_batch.numpy())
        
        # Compute MAPE on validation set
        val_mape = mean_absolute_percentage_error(val_targets, val_predictions)
        
        # Save the best model
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_model_state = model.state_dict()
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAPE: {val_mape:.4f}')
    
    # Load the best model
    model.load_state_dict(best_model_state)
    return model, best_val_mape


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out
    
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_gru, _ = self.gru(x)
        out = self.fc(h_gru[:, -1, :])
        return out
    
# lr_model = LinearRegression()

# dt_model = DecisionTreeRegressor()

# xgb_model = XGBRegressor()
