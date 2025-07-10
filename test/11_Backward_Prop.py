import numpy as np

# --- Activation Functions & Derivatives ---
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(output): return output * (1 - output) # output = sigmoid(z)

def relu(x): return np.maximum(0, x)
def relu_derivative(output): return np.where(output > 0, 1, 0) # output = relu(z)

def linear(x): return x
def linear_derivative(output): return np.ones_like(output) # output = linear(z)


# --- Loss Function ---
def mse_loss(y_true, y_pred): return np.mean((y_true - y_pred)**2)

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

# MLP class
class MLP:
    def __init__(self): self.layers = []
    def add_layer(self, layer): self.layers.append(layer)
    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers: current_input = layer.forward(current_input)
        return current_input

if __name__ == "__main__":
    # Single layer: 2 inputs, 3 neurons, sigmoid activation
    layer = DenseLayer(n_inputs=2, n_neurons=3, activation_fn_name='sigmoid')
    
    X_sample = np.array([[0.5, -0.2]]) # 1 sample, 2 features
    print(f"Input X: {X_sample}")
    
    # Forward pass
    layer_output = layer.forward(X_sample)
    print(f"Layer output (after sigmoid): {layer_output}")

    # Assume a dummy gradient from a hypothetical next layer or loss function
    # This is d(Loss)/d(layer_output)
    # Shape must match layer_output: (n_samples, n_neurons_in_layer)
    dummy_d_loss_wrt_layer_output = np.array([[0.1, -0.2, 0.05]]) 
    print(f"Dummy dL/d(layer_output): {dummy_d_loss_wrt_layer_output}")

    # Backward pass
    d_loss_wrt_input = layer.backward(dummy_d_loss_wrt_layer_output)

    print(f"\nCalculated Gradients:")
    print(f"  dL/d_weights (shape {layer.d_weights.shape}):\n{layer.d_weights}")
    print(f"  dL/d_biases (shape {layer.d_biases.shape}):\n{layer.d_biases}")
    print(f"  dL/d_inputs (to pass to prev layer) (shape {d_loss_wrt_input.shape}):\n{d_loss_wrt_input}")