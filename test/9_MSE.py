import numpy as np

# --- Activation Functions ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)
def linear(x): return x

# --- Loss Function ---
def mse_loss(y_true, y_pred):
    """
    Calculate Mean Squared Error loss.
    y_true: NumPy array of true target values.
    y_pred: NumPy array of predicted values.
    """
    return np.mean((y_true - y_pred) ** 2)

class DenseLayer:
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
        self.output = None

        if activation_fn_name == 'sigmoid': self.activation_fn = sigmoid
        elif activation_fn_name == 'relu': self.activation_fn = relu
        elif activation_fn_name == 'linear': self.activation_fn = linear
        else: raise ValueError(f"Unsupported activation: {activation_fn_name}")

    def forward(self, inputs):
        self.inputs = inputs 
        weighted_sum = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation_fn(weighted_sum)
        return self.output

class MLP:
    def __init__(self): self.layers = []
    def add_layer(self, layer): self.layers.append(layer)
    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

if __name__ == "__main__":
    # --- Single-sample demo ---
    X_sample = np.array([[0.1, 0.2, -0.1]])
    y_true_sample = np.array([[0.8]])

    mlp = MLP()
    mlp.add_layer(DenseLayer(3, 5, 'relu'))
    mlp.add_layer(DenseLayer(5, 1, 'linear'))

    print("MLP Architecture:")
    for i, layer in enumerate(mlp.layers):
        print(f"  Layer {i+1}: {layer.n_inputs} inputs, {layer.n_neurons} neurons, Activation: {layer.activation_fn_name}")

    y_pred_sample = mlp.forward(X_sample)
    print("\n--- Single Sample ---")
    print(f"Input X: {X_sample}")
    print(f"True y: {y_true_sample}")
    print(f"Predicted y: {y_pred_sample}")
    loss_sample = mse_loss(y_true_sample, y_pred_sample)
    print(f"Mean Squared Error Loss (single sample): {loss_sample:.4f}")

    # --- Batch demo ---
    X_batch = np.array([
        [0.1, 0.2, -0.1],
        [0.5, -0.3, 0.8],
        [0.0, 0.7, 0.2]
    ])
    y_true_batch = np.array([
        [0.8],
        [0.2],
        [0.5]
    ])

    y_pred_batch = mlp.forward(X_batch)
    loss_batch = mse_loss(y_true_batch, y_pred_batch)

    print("\n--- Batch Sample ---")
    print(f"Batch Input X shape: {X_batch.shape}")
    print(f"Batch True y shape: {y_true_batch.shape}")
    print(f"Batch Predicted y shape: {y_pred_batch.shape}")
    print(f"Predicted y (batch):\n{y_pred_batch}")
    print(f"Mean Squared Error Loss (batch): {loss_batch:.4f}")