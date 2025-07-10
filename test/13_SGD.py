import numpy as np
from sklearn.datasets import load_diabetes # Import diabetes dataset

# --- Activation Functions & Derivatives (from previous unit) ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(output): return output * (1 - output)
def relu(x): return np.maximum(0, x)
def relu_derivative(output): return np.where(output > 0, 1, 0)
def linear(x): return x
def linear_derivative(output): return np.ones_like(output)

# --- Loss Function & Derivative (from previous unit) ---
def mse_loss(y_true, y_pred): return np.mean((y_true - y_pred)**2)
def mse_loss_derivative(y_true, y_pred): return 2 * (y_pred - y_true) / y_true.shape[0]

class DenseLayer: # (Unchanged from previous unit)
    def __init__(self, n_inputs, n_neurons, activation_fn_name='sigmoid',
                 weight_init_strategy='random_scaled', weight_init_scale=0.01):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation_fn_name = activation_fn_name
        self.weight_init_strategy = weight_init_strategy
        self.weight_init_scale = weight_init_scale 

        if weight_init_strategy == 'random_scaled':
            self.weights = np.random.randn(n_inputs, n_neurons) * weight_init_scale
        elif weight_init_strategy == 'xavier_normal':
            stddev = np.sqrt(2.0 / (self.n_inputs + self.n_neurons))
            self.weights = np.random.randn(n_inputs, n_neurons) * stddev
        elif weight_init_strategy == 'he_uniform':
            limit = np.sqrt(6.0 / self.n_inputs)
            self.weights = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        else:
            raise ValueError(f"Unsupported weight initialization strategy: {weight_init_strategy}")

        self.biases = np.zeros((1, n_neurons))

        if activation_fn_name == 'sigmoid': 
            self.activation_fn = sigmoid
            self.activation_derivative_fn = sigmoid_derivative
        elif activation_fn_name == 'relu':
            self.activation_fn = relu
            self.activation_derivative_fn = relu_derivative
        elif activation_fn_name == 'linear':
            self.activation_fn = linear
            self.activation_derivative_fn = linear_derivative
        else: raise ValueError(f"Unsupported activation: {activation_fn_name}")
        
        self.inputs = None; self.z = None; self.output = None
        self.d_weights = None; self.d_biases = None
    def forward(self, inputs):
        self.inputs = inputs; self.z = np.dot(inputs, self.weights) + self.biases; self.output = self.activation_fn(self.z)
        return self.output
    def backward(self, d_loss_wrt_layer_output):
        d_activation_output = d_loss_wrt_layer_output * self.activation_derivative_fn(self.output)
        self.d_weights = np.dot(self.inputs.T, d_activation_output)
        self.d_biases = np.sum(d_activation_output, axis=0, keepdims=True)
        return np.dot(d_activation_output, self.weights.T)

class MLP: # (Unchanged from previous unit)
    def __init__(self): self.layers = []
    def add_layer(self, layer): self.layers.append(layer)
    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers: current_input = layer.forward(current_input)
        return current_input
    def backward(self, d_loss_wrt_prediction):
        current_d_loss = d_loss_wrt_prediction
        for layer in reversed(self.layers): current_d_loss = layer.backward(current_d_loss)

# --- Optimizer ---
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    def update(self, layer):
        """Update layer's weights and biases using its stored gradients."""
        if layer.d_weights is not None and layer.d_biases is not None:
            layer.weights -= self.learning_rate * layer.d_weights
            layer.biases -= self.learning_rate * layer.d_biases
        else:
            print("Warning: Gradients not found for a layer. Skipping update.")


if __name__ == "__main__":
    # Sample data: Diabetes dataset for regression (minimal preprocessing)
    diabetes = load_diabetes()
    X_train, y_train = diabetes.data, diabetes.target
    
    # Reshape y_train to be a column vector (N, 1)
    y_train = y_train.reshape(-1, 1)
    
    # For simplicity, we will not scale the data in this unit.
    # This might require adjusting learning rate and epochs.
    n_samples, n_features = X_train.shape

    mlp = MLP()
    # Hidden layer: n_features inputs, e.g., 10 neurons, ReLU activation
    mlp.add_layer(DenseLayer(n_features, 10, 'relu', weight_init_scale=0.1))
    # Output layer: 10 inputs (from hidden layer), 1 neuron (regression), Linear activation
    mlp.add_layer(DenseLayer(10, 1, 'linear', weight_init_scale=0.1)) 
    
    optimizer = SGD(learning_rate=0.002) # Small learning rate
    epochs = 100 # Number of epochs
    batch_size = 32  # Define batch size for SGD
    num_batches = (n_samples + batch_size - 1) // batch_size  # Total number of batches in each epoch

    print(f"Training MLP for Diabetes regression for {epochs} epochs with LR={optimizer.learning_rate}, batch_size={batch_size}")
    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")

    # TODO: Implement the full training loop below.
    for epoch in range(epochs):
        # 1. Shuffle the data at the beginning of each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # 2. Initialize total loss
        total_loss = 0

        # 3. Process data in mini-batches
        for i in range(0, n_samples, batch_size):
            # 4. Extract current mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # 5. Forward pass
            y_pred = mlp.forward(X_batch)

            # 6. Calculate loss and accumulate it
            loss = mse_loss(y_batch, y_pred)
            total_loss += loss

            # 7. Compute gradient of loss
            d_loss_wrt_pred = mse_loss_derivative(y_batch, y_pred)

            # 8. Backward pass
            mlp.backward(d_loss_wrt_pred)

            # 9. Update weights
            for layer in mlp.layers:
                optimizer.update(layer)

        # 10. Compute average loss for this epoch
        avg_loss = total_loss / num_batches

        # 11. Print progress every 10% of total epochs or on first epoch
        if (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs}, Loss: {avg_loss:.2f}")

    print("\n--- Training Over ---")
    final_preds = mlp.forward(X_train)
    final_loss = mse_loss(y_train, final_preds)
    print(f"Final MSE Loss on training data: {final_loss:.2f}")

    print("\nSample Predictions:")
    for i in range(min(5, n_samples)): # Print first 5 samples
        print(f"  True: {y_train[i][0]:8.2f}, Predicted: {final_preds[i][0]:8.2f}")