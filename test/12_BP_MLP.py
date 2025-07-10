import numpy as np

# --- Activation Functions & Derivatives (from previous unit) ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(output): return output * (1 - output)
def relu(x): return np.maximum(0, x)
def relu_derivative(output): return np.where(output > 0, 1, 0)
def linear(x): return x
def linear_derivative(output): return np.ones_like(output)


# --- Loss Function & Derivative ---
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_loss_derivative(y_true, y_pred):
    """Derivative of MSE loss w.r.t. y_pred."""
    return 2 * (y_pred - y_true) / y_true.shape[0] # Normalize by batch size

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
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases # Pre-activation output
        self.output = self.activation_fn(self.z)
        return self.output

    def backward(self, d_loss_wrt_layer_output):
        d_activation_output = d_loss_wrt_layer_output * self.activation_derivative_fn(self.output)
        self.d_weights = np.dot(self.inputs.T, d_activation_output)
        self.d_biases = np.sum(d_activation_output, axis=0, keepdims=True)
        d_loss_wrt_prev_layer_output = np.dot(d_activation_output, self.weights.T)
        return d_loss_wrt_prev_layer_output


class MLP:
    def __init__(self): self.layers = []
    def add_layer(self, layer): self.layers.append(layer)
    
    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers: current_input = layer.forward(current_input)
        return current_input

    def backward(self, d_loss_wrt_prediction):
        current_d_loss = d_loss_wrt_prediction
        # Propagate backward through layers in reverse order
        for layer in reversed(self.layers):
            current_d_loss = layer.backward(current_d_loss)


if __name__ == "__main__":
    X_train = np.array([[0.1, 0.2], [0.3, 0.4], [-0.1, 0.5]]) # 3 samples, 2 features
    y_train = np.array([[0.5], [0.8], [0.2]])              # 3 samples, 1 output (regression)

    mlp = MLP()
    mlp.add_layer(DenseLayer(2, 4, 'relu'))
    mlp.add_layer(DenseLayer(4, 1, 'linear')) # Output for regression

    # 1. Forward pass
    y_pred = mlp.forward(X_train)
    
    # 2. Calculate loss
    loss = mse_loss(y_train, y_pred)
    print(f"Input X_train (shape {X_train.shape}):\n{X_train}")
    print(f"True y_train (shape {y_train.shape}):\n{y_train}")
    print(f"Predicted y_pred (shape {y_pred.shape}):\n{y_pred}")
    print(f"Initial MSE Loss: {loss:.4f}")

    # 3. Calculate derivative of loss w.r.t. prediction
    d_loss_wrt_pred = mse_loss_derivative(y_train, y_pred)
    print(f"\nDerivative of Loss w.r.t. Prediction (dL/dy_pred) (shape {d_loss_wrt_pred.shape}):\n{d_loss_wrt_pred}")

    # 4. Backward pass through MLP
    mlp.backward(d_loss_wrt_pred)

    print("\nGradients after backpropagation:")
    for i, layer in enumerate(mlp.layers):
        print(f"  Layer {i+1} ({layer.activation_fn_name}):")
        print(f"    dL/d_weights (shape {layer.d_weights.shape}):\n{layer.d_weights[:2,:2]}...") # Excerpt
        print(f"    dL/d_biases (shape {layer.d_biases.shape}):\n{layer.d_biases}")