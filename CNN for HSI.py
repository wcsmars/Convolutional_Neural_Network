import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim

# Download the data
ticker_symbol = "^HSI"
start_date = '1900-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)

data = data.dropna().reset_index(drop=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume']
all_data = data[features].values

# Initialize the scaler
scaler = MinMaxScaler()
scaled_all = scaler.fit_transform(all_data)

# Split scaled data into features (X_features) and target (y_close)
X_features = scaled_all[:, [0, 1, 2, 4]]
y_close = scaled_all[:, 3]

# Set the lookback window size
lookback = 40
X, y = [], []
for i in range(len(X_features) - lookback):
    X.append(X_features[i:i+lookback])
    # Now the target is the close price at position i+lookback
    y.append(y_close[i + lookback])
X, y = np.array(X), np.array(y)

train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Convert the data to PyTorch tensors
X_train, X_val, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_val), torch.FloatTensor(X_test)
y_train, y_val, y_test = torch.FloatTensor(y_train), torch.FloatTensor(y_val), torch.FloatTensor(y_test)

# Create DataLoader objects for easier batch processing
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# Loader without shuffling for directional accuracy calculation
train_eval_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the EarlyStopping class
class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss >= self.best_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), 'best_model.pt')
        print('Checkpoint saved')

class CNNModel(nn.Module):
    def __init__(self, dropout_rate=0):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1) # 4 features
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * lookback, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # Transpose the input tensor dimensions
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout3(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

def calc_up_down_accuracy(predictions, targets):
    """Return directional accuracy for ordered predictions and targets."""
    up_down_pred = torch.sign(predictions[1:] - predictions[:-1])
    up_down_target = torch.sign(targets[1:] - targets[:-1])
    return (up_down_pred == up_down_target).float().mean().item()


# Initialize counters for training and validation up/down accuracy
train_up_down_accuracy_list = []
val_up_down_accuracy_list = []



# Instantiate the model, criterion, optimizer, and early stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNNModel(dropout_rate=0.1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
early_stopping = EarlyStopping(patience=50)

# Train the model and store the loss values
train_losses = []
val_losses = []

num_epochs = 300

# Add scheduler.step() to the training loop
for epoch in range(num_epochs):
    # Training
    cnn_model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Directional accuracy on the training set in temporal order
    cnn_model.eval()
    train_preds = []
    train_targets = []
    with torch.no_grad():
        for t_inputs, t_targets in train_eval_loader:
            t_inputs, t_targets = t_inputs.to(device), t_targets.to(device)
            t_outputs = cnn_model(t_inputs)
            train_preds.append(t_outputs.squeeze())
            train_targets.append(t_targets)
    train_preds = torch.cat(train_preds)
    train_targets = torch.cat(train_targets)
    train_up_down_accuracy = calc_up_down_accuracy(train_preds, train_targets)
    train_up_down_accuracy_list.append(train_up_down_accuracy)
    
    # Validation
    cnn_model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = cnn_model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()

            val_preds.append(outputs.squeeze())
            val_targets.append(targets)

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    val_up_down_accuracy = calc_up_down_accuracy(val_preds, val_targets)
    val_up_down_accuracy_list.append(val_up_down_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, Val Loss: {val_loss}, Train Up/Down Acc: {train_up_down_accuracy}, Val Up/Down Acc: {val_up_down_accuracy}")

    early_stopping(val_loss, cnn_model)
    if early_stopping.stop:
        print("Early stopping triggered")
        break

    # Update the learning rate scheduler
    scheduler.step()

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Obtain model predictions on the test set
cnn_model.load_state_dict(torch.load('best_model.pt'))
cnn_model.eval()
predictions = []
ground_truth = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = cnn_model(inputs)
        predictions.extend(outputs.squeeze().cpu().numpy())
        ground_truth.extend(targets.cpu().numpy())

# Plot the predictions against the ground truth
plt.figure(figsize=(10, 5))
plt.plot(predictions, label='Predictions', color='r')
plt.plot(ground_truth, label='Ground Truth', color='g', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('HSI Price')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(predictions[0:20], label='Predictions', color='r')
plt.plot(ground_truth[0:20], label='Ground Truth', color='g', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('HSI Price')
plt.legend()
plt.show()

# Run the model on the validation dataset and store the predictions
predictions = []
ground_truth = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = cnn_model(inputs)
        predictions.extend(outputs.squeeze().tolist())
        ground_truth.extend(targets.tolist())

predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

# After training and getting predictions, suppose you have:
# predictions and ground_truth as arrays of prices, both are scaled versions.

# Inverse transform predictions and ground_truth
# Create dummy arrays to inverse transform since you only scaled entire feature sets.
dummy_pred = np.zeros((len(predictions), len(features)))
dummy_ground = np.zeros((len(ground_truth), len(features)))

# Insert predictions and ground truth into the correct column (close price is at index 3)
dummy_pred[:, 3] = predictions
dummy_ground[:, 3] = ground_truth

original_pred = scaler.inverse_transform(dummy_pred)[:, 3]
original_ground = scaler.inverse_transform(dummy_ground)[:, 3]

# Compute predicted direction
# predicted_up_down[i] = 1 if model predicts upward movement from day i to i+1, else -1
predicted_up_down = np.sign(original_pred[1:] - original_pred[:-1])

# Compute actual next-day returns
# daily_returns[i] = price[i+1] - price[i]
daily_returns = original_ground[1:] - original_ground[:-1]

# Now simulate the strategy
profits = []
for i in range(len(daily_returns)):
    if predicted_up_down[i] == 1:
        # Model predicts up, so go long on day i and sell on day i+1
        # Profit = actual next-day return
        profit = daily_returns[i]
    else:
        # Model predicts down, so short on day i and buy back on day i+1
        # Profit = -(actual next-day return), since a price drop yields profit
        profit = -daily_returns[i]
    profits.append(profit)

# Convert profits to a NumPy array if needed
profits = np.array(profits)

# Calculate cumulative returns
cumulative_returns = np.cumsum(profits)

# Print summary statistics
print(f"Average Daily Profit: {np.mean(profits)}")
print(f"Total Profit Over Period: {np.sum(profits)}")
print(
    f"Winning Rate (Directional Accuracy): {np.mean(predicted_up_down == np.sign(daily_returns))}"
)
# Plot the cumulative returns curve
plt.figure(figsize=(10, 5))
plt.plot(cumulative_returns, label='Strategy Cumulative PnL')
plt.xlabel('Days')
plt.ylabel('Cumulative Profit')
plt.legend()
plt.show()
