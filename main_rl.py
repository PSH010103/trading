import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pandas as pd

from sklearn.preprocessing import StandardScaler

class TimeSeriesDQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1):  # Reduced hidden size
        super(TimeSeriesDQNNetwork, self).__init__()
        
        # Smaller architecture with dropout for regularization
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout to prevent overfitting
            nn.Linear(hidden_size, hidden_size//2),  # Reduced second layer size
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size),
            nn.Tanh()  # Bound output between -1 and 1
        )
        
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_size, output_size=1, hidden_size=64, learning_rate=1e-4,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.batch_size = batch_size
        
        # Networks
        self.policy_net = TimeSeriesDQNNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_net = TimeSeriesDQNNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Action bounds
        self.action_bounds = (-1.0, 1.0)
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                # Add batch dimension
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.policy_net(state).cpu().numpy()[0]
                return np.clip(action, self.action_bounds[0], self.action_bounds[1])
        else:
            return np.random.uniform(self.action_bounds[0], self.action_bounds[1], 1)
    
    def compute_reward(self, action, target):
        """Compute bounded reward"""
        # Extract single values from numpy arrays if necessary
        action_val = action[0] if isinstance(action, np.ndarray) else action
        target_val = target[0] if isinstance(target, np.ndarray) else target
        error = abs(float(action_val) - float(target_val))
        return -min(error, 1.0)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).reshape(-1, 1).to(self.device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states)
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            target_q_values = rewards + self.gamma * next_q_values
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    def update_target_network(self):
        """Update target network by copying the weights of the policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_dqn_predictor(name, epochs=1000, test_size=0.2):
    # Load and preprocess data
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(name, test_size)
    input_size = X_train.shape[1]
    
    # Initialize agent
    agent = DQNAgent(input_size=input_size)
    
    train_losses = []
    test_losses = []
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Training loop
        for i in range(len(X_train)):
            state = X_train[i]
            action = agent.select_action(state)
            reward = agent.compute_reward(action, y_train[i])
            next_state = X_train[(i + 1) % len(X_train)]
            
            # Store experience
            agent.memory.push(state, action, reward, next_state)
            
            # Train the agent
            loss = agent.train_step()
            if loss is not None:
                epoch_losses.append(loss)
        
        # Update target network periodically
        if epoch % 10 == 0:
            agent.update_target_network()
            
            # Evaluate on test set
            test_predictions = predict(agent, X_test)
            test_loss = np.mean((test_predictions - y_test.flatten()) ** 2)
            test_losses.append(test_loss)
        
        # Log progress
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, "
                      f"Test Loss = {test_losses[-1] if test_losses else 'N/A':.4f}, "
                      f"Epsilon = {agent.epsilon:.4f}")
    
    return agent, (train_losses, test_losses)

def load_and_preprocess_data(name, test_size=0.2):
    """Load and preprocess data from CSV files"""
    # Load data
    df_x = pd.read_csv(f'./processed_data/{name}_X.csv', index_col=0)
    df_y = pd.read_csv(f'./processed_data/{name}_y.csv', index_col=0)
    
    # Convert to numpy arrays
    X = df_x.values
    y = df_y.values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    split_idx = int(len(X_scaled) * (1 - test_size))
    X_train = X_scaled[:split_idx]
    y_train = y[:split_idx]
    X_test = X_scaled[split_idx:]
    y_test = y[split_idx:]
    
    return X_train, y_train, X_test, y_test, scaler

def train_dqn_predictor(name, epochs=1000, test_size=0.2):
    # Load and preprocess data
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(name, test_size)
    input_size = X_train.shape[1]
    
    # Initialize agent
    agent = DQNAgent(input_size=input_size)
    
    train_losses = []
    test_losses = []
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Training loop
        for i in range(len(X_train)):
            state = X_train[i]
            action = agent.select_action(state)
            reward = agent.compute_reward(action, y_train[i])  # Use new reward function
            next_state = X_train[(i + 1) % len(X_train)]
            
            # Store experience
            agent.memory.push(state, action, reward, next_state)
            
            # Train the agent
            loss = agent.train_step()
            if loss is not None:
                epoch_losses.append(loss)
        
        # Update target network periodically
        if epoch % 10 == 0:
            agent.update_target_network()
            
            # Evaluate on test set
            test_predictions = predict(agent, X_test)
            test_loss = np.mean((test_predictions - y_test.flatten()) ** 2)
            test_losses.append(test_loss)
        
        # Log progress
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, "
                      f"Test Loss = {test_losses[-1] if test_losses else 'N/A':.4f}, "
                      f"Epsilon = {agent.epsilon:.4f}")
    
    return agent, (train_losses, test_losses)

def predict(agent, X):
    """Make predictions using the trained agent"""
    predictions = []
    with torch.no_grad():
        for state in X:
            state_tensor = torch.FloatTensor(state).to(agent.device)
            prediction = agent.policy_net(state_tensor).cpu().numpy()
            predictions.append(prediction[0])
    return np.array(predictions)

def evaluate_model(agent, X_test, y_test):
    """Evaluate the model performance"""
    predictions = predict(agent, X_test)
    mse = np.mean((predictions - y_test.flatten()) ** 2)
    mae = np.mean(np.abs(predictions - y_test.flatten()))    
    mape = np.mean(np.abs((predictions - y_test.flatten()) / y_test.flatten())) * 100  # MAPE in percentage
    
    return {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions
    }
    
    
# Train the model
name = "AAPL"  
test_size = 0.2  # 20% test set

# Train the model
agent, (train_losses, test_losses) = train_dqn_predictor(
    name, 
    epochs=10,
    test_size=test_size
)

# Load data for final evaluation
X_train, y_train, X_test, y_test, _ = load_and_preprocess_data(name, test_size)

# Evaluate the model
evaluation_results = evaluate_model(agent, X_test, y_test)
print(f"Test MSE: {evaluation_results['mse']:.4f}")
print(f"Test MAE: {evaluation_results['mae']:.4f}")
print(f"Test MAPE: {evaluation_results['mape']:.4f}")