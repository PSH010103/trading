import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TradingDQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(TradingDQNNetwork, self).__init__()
        
        # Three actions: Buy (0), Hold (1), Sell (2)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 3)  # 3 actions: buy, hold, sell
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states),
                np.array(dones, dtype=np.bool_))
    
    def __len__(self):
        return len(self.buffer)

class TradingDQNAgent:
    def __init__(self, input_size, hidden_size=64, learning_rate=1e-4,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.batch_size = batch_size
        
        # Networks
        self.policy_net = TradingDQNNetwork(input_size, hidden_size).to(self.device)
        self.target_net = TradingDQNNetwork(input_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Trading state
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()  # 0: Buy, 1: Hold, 2: Sell
        else:
            return random.randint(0, 2)
    
    def compute_reward(self, action, price, next_price):
        """
        Compute reward based on trading action and price change
        Returns: reward, done
        """
        reward = 0
        done = False
        
        # Update position based on action
        if action == 0:  # Buy
            if self.position <= 0:  # If not already long
                self.position = 1
                self.entry_price = price
        elif action == 2:  # Sell
            if self.position >= 0:  # If not already short
                self.position = -1
                self.entry_price = price
                
        # Calculate reward based on position and price change
        if self.position != 0:
            price_change = (next_price - price) / price
            reward = price_change * self.position  # Positive for correct direction
            
        return reward, done
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
def load_and_preprocess_data(name, test_size=0.2):
    """Load and preprocess train/test data directly from CSV files"""
    # Load train data
    train_x = pd.read_csv(f'./train_X.csv', index_col=0)
    train_y = pd.read_csv(f'./train_y.csv', index_col=0)
    
    # Load test data
    test_x = pd.read_csv(f'./test_X.csv', index_col=0)
    test_y = pd.read_csv(f'./test_y.csv', index_col=0)
    
    # Convert to numpy arrays
    X_train = train_x.values
    y_train = train_y.values
    X_test = test_x.values
    y_test = test_y.values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data only
    X_test_scaled = scaler.transform(X_test)  # Transform test data using training data parameters
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler
def train_trading_agent(name, epochs=1000, test_size=0.2):
    # Load and preprocess data
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(name, test_size)
    input_size = X_train.shape[1]
    
    # Initialize agent
    agent = TradingDQNAgent(input_size=input_size)
    
    train_profits = []
    test_profits = []
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples")
    
    for epoch in range(epochs):
        epoch_profits = []
        agent.position = 0  # Reset position at start of epoch
        
        # Training loop
        for i in range(len(X_train)-1):
            state = X_train[i]
            current_price = y_train[i][0]
            next_price = y_train[i+1][0]
            
            action = agent.select_action(state)
            reward, done = agent.compute_reward(action, current_price, next_price)
            next_state = X_train[i + 1]
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train the agent
            loss = agent.train_step()
            epoch_profits.append(reward)
        
        # Update target network periodically
        if epoch % 10 == 0:
            agent.update_target_network()
            
            # Evaluate on test set
            test_returns = evaluate_trading(agent, X_test, y_test)
            test_profits.append(np.mean(test_returns))
        
        # Log progress
        if epoch_profits:
            avg_profit = np.mean(epoch_profits)
            train_profits.append(avg_profit)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Profit = {avg_profit:.4f}, "
                      f"Test Profit = {test_profits[-1] if test_profits else 'N/A':.4f}, "
                      f"Epsilon = {agent.epsilon:.4f}")
    
    return agent, (train_profits, test_profits)

def evaluate_trading(agent, X_test, y_test):
    """Evaluate the trading strategy"""
    returns = []
    agent.position = 0  # Reset position
    
    for i in range(len(X_test)-1):
        state = X_test[i]
        current_price = y_test[i][0]
        next_price = y_test[i+1][0]
        
        with torch.no_grad():
            action = agent.select_action(state)
            reward, _ = agent.compute_reward(action, current_price, next_price)
            returns.append(reward)
    
    return np.array(returns)

def calculate_trading_metrics(returns):
    """Calculate trading performance metrics"""
    total_return = np.prod(1+returns)
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    max_drawdown = np.min(np.minimum.accumulate(np.cumsum(returns)))
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'returns': returns
    }
    
    
    # Train the model
agent, (train_profits, test_profits) = train_trading_agent(
    name="AAPL", 
    epochs=50,
    test_size=0.2
)

# Evaluate trading performance
X_train, y_train, X_test, y_test, _ = load_and_preprocess_data("AAPL", 0.2)
test_returns = evaluate_trading(agent, X_test, y_test)
metrics = calculate_trading_metrics(test_returns)

print(f"Total Return: {metrics['total_return']:.2f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2f}")