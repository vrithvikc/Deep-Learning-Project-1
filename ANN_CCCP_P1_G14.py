import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_excel('Folds5x2_pp.xlsx')

# Normalize function
def normalize_data(data):
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    # Map to [-0.9, 0.9]
    normalized_data = 1.8 * (data - min_val) / (max_val - min_val) - 0.9
    return normalized_data

# Splitting data into training, validation, and test sets
def split_data(data):
    np.random.seed(0)  # For reproducibility
    indices = np.random.permutation(len(data))  # Shuffle indices
    shuffled_data = data.values[indices]  # Shuffle the data using the shuffled indices
    train_size = int(len(shuffled_data) * 0.72)
    val_size = int(len(shuffled_data) * 0.18)
    
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    return train_data, val_data, test_data

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
    epsilon = 0.001  # Small positive number to avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)))  # MAPE in percentage

# ANN Class with Two Hidden Layers (ReLU for hidden layers, Sigmoid for output layer)
class ANN:
    def _init_(self, input_size, hidden_size, output_size, learning_rate=0.001, momentum=0.9, l2_lambda=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        
        # Weight initialization for 2 hidden layers
        self.weights_input_hidden1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.weights_hidden1_hidden2 = np.random.uniform(-1, 1, (self.hidden_size, self.hidden_size))
        self.weights_hidden2_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        
        self.bias_hidden1 = np.zeros((1, self.hidden_size))
        self.bias_hidden2 = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

        # Momentum terms
        self.velocity_input_hidden1 = np.zeros_like(self.weights_input_hidden1)
        self.velocity_hidden1_hidden2 = np.zeros_like(self.weights_hidden1_hidden2)
        self.velocity_hidden2_output = np.zeros_like(self.weights_hidden2_output)

    def forward(self, X):
        # Apply ReLU for the first hidden layer
        self.hidden1_input = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden1_output = relu(self.hidden1_input)
        
        # Apply ReLU for the second hidden layer
        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden2_output = relu(self.hidden2_input)
        
        # Apply sigmoid for the output layer
        self.final_input = np.dot(self.hidden2_output, self.weights_hidden2_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        # Compute gradients
        output_error = y - output  # Error at output
        output_delta = output_error * sigmoid_derivative(output)  # Delta at output
        
        hidden2_error = np.dot(output_delta, self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * relu_derivative(self.hidden2_output)
        
        hidden1_error = np.dot(hidden2_delta, self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * relu_derivative(self.hidden1_output)
        
        # Update weights and biases with momentum and L2 Regularization
        # Update weights_hidden2_output and bias_output
        grad_weights_hidden2_output = np.dot(self.hidden2_output.T, output_delta)
        grad_bias_output = np.sum(output_delta, axis=0, keepdims=True)
        
        self.velocity_hidden2_output = self.momentum * self.velocity_hidden2_output + \
                                       self.learning_rate * grad_weights_hidden2_output - \
                                       self.l2_lambda * self.weights_hidden2_output
        self.weights_hidden2_output += self.velocity_hidden2_output
        self.bias_output += self.learning_rate * grad_bias_output

        # Update weights_hidden1_hidden2 and bias_hidden2
        grad_weights_hidden1_hidden2 = np.dot(self.hidden1_output.T, hidden2_delta)
        grad_bias_hidden2 = np.sum(hidden2_delta, axis=0, keepdims=True)
        
        self.velocity_hidden1_hidden2 = self.momentum * self.velocity_hidden1_hidden2 + \
                                         self.learning_rate * grad_weights_hidden1_hidden2 - \
                                         self.l2_lambda * self.weights_hidden1_hidden2
        self.weights_hidden1_hidden2 += self.velocity_hidden1_hidden2
        self.bias_hidden2 += self.learning_rate * grad_bias_hidden2

        # Update weights_input_hidden1 and bias_hidden1
        grad_weights_input_hidden1 = np.dot(X.T, hidden1_delta)
        grad_bias_hidden1 = np.sum(hidden1_delta, axis=0, keepdims=True)
        
        self.velocity_input_hidden1 = self.momentum * self.velocity_input_hidden1 + \
                                       self.learning_rate * grad_weights_input_hidden1 - \
                                       self.l2_lambda * self.weights_input_hidden1
        self.weights_input_hidden1 += self.velocity_input_hidden1
        self.bias_hidden1 += self.learning_rate * grad_bias_hidden1

    def train(self, X, y, X_val, y_val, X_test, y_test, epochs=1000, batch_size=1):
        train_losses = []
        val_losses = []
        train_mapes = []
        val_mapes = []
        
        for epoch in range(epochs):
            # Shuffle the training data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.forward(X_batch)
                loss = np.mean(np.square(output - y_batch))

                self.backward(X_batch, y_batch, output)

            # Compute MAPE
            train_pred = self.forward(X)
            train_mape = mape(y, train_pred)
            val_pred = self.forward(X_val)
            val_mape = mape(y_val, val_pred)

            # Compute losses
            train_loss = np.mean(np.square(y - train_pred))
            val_loss = np.mean(np.square(y_val - val_pred))
            
            # Store histories
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_mapes.append(train_mape)
            val_mapes.append(val_mape)

            # Logging every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train MAPE: {train_mape:.2f}%, Val MAPE: {val_mape:.2f}%')

        # Evaluate on test data after full training
        test_output = self.predict(X_test)
        test_loss = np.mean(np.square(y_test - test_output))
        test_mape = mape(y_test, test_output)
        print(f"Final Test Loss: {test_loss:.4f}, Test MAPE: {test_mape:.2f}%")
        self.plot_results(train_mapes, val_mapes)

        return train_losses, val_losses, train_mapes, val_mapes

    def predict(self, X):
        return self.forward(X)

    def plot_results(self, train_mapes, val_mapes):
        # Plot training and validation MAPE after full training
        plt.figure(figsize=(12, 5))
        plt.plot(train_mapes, label='Training MAPE')
        plt.plot(val_mapes, label='Validation MAPE')
        plt.xlabel('Epochs')
        plt.ylabel('MAPE (%)')
        plt.title('Training vs Validation MAPE after 1000 Epochs')
        plt.legend()
        plt.show()

# Main execution
if _name_ == "_main_":
    # Normalize the data
    normalized_data = normalize_data(data)

    # Split the data
    train_data, val_data, test_data = split_data(normalized_data)

    # Separate input and output
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].reshape(-1, 1)
    
    X_val = val_data[:, :-1]
    y_val = val_data[:, -1].reshape(-1, 1)
    
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].reshape(-1, 1)

    # Create ANN instance
    input_size = X_train.shape[1]
    hidden_size = 10  # Number of neurons in each hidden layer
    output_size = 1

    ann = ANN(input_size, hidden_size, output_size)

    # Train the ANN
    train_losses, val_losses, train_mapes, val_mapes = ann.train(X_train, y_train, X_val, y_val, X_test, y_test, epochs=1000, batch_size=256)
