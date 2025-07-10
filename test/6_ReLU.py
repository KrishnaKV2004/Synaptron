import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation function."""
    return np.maximum(x, 0)

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation_fn_name='sigmoid'):
        self.weights = np.random.rand(n_inputs, n_neurons) * 0.1 
        self.biases = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.output = None
        self.activation_fn_name = activation_fn_name

        if activation_fn_name == 'sigmoid':
            self.activation_fn = sigmoid
        elif activation_fn_name == 'relu':
            self.activation_fn = relu
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn_name}")

    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation_fn(weighted_sum)
        return self.output

class MLP:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

if __name__ == "__main__":
    X_sample = np.array([[-1.0, 0.5, 2.0, -0.1],
                         [1.0, 0.5, 2.0, 0.1]]) 
    print(f"Input X (shape {X_sample.shape}):\n{X_sample}")

    mlp_relu = MLP()
    mlp_relu.add_layer(DenseLayer(n_inputs=4, n_neurons=5, activation_fn_name='relu'))
    mlp_relu.add_layer(DenseLayer(n_inputs=5, n_neurons=3, activation_fn_name='sigmoid'))
    mlp_relu.add_layer(DenseLayer(n_inputs=3, n_neurons=1, activation_fn_name='sigmoid'))

    print(f"\nMLP created with {len(mlp_relu.layers)} layers and mixed activations.")
    for i, layer in enumerate(mlp_relu.layers):
        print(f"  Layer {i+1}: {layer.n_inputs} inputs, {layer.n_neurons} neurons, Activation: {layer.activation_fn_name}")

    output_relu = mlp_relu.forward(X_sample)
    print(f"\nOutput of the MLP (shape {output_relu.shape}):\n{output_relu}")
    
    X_negative_heavy = np.array([[-1.0, -0.5, -2.0, -0.1],
                                 [-2.0, -0.1, -3.0, -1.1]])
    print(f"\nInput with mostly negative values (shape {X_negative_heavy.shape}):\n{X_negative_heavy}")
    # Store first layer to inspect its output
    first_layer_output_before_forward = mlp_relu.layers[0].output
    output_negative_heavy = mlp_relu.forward(X_negative_heavy)
    print(f"Output for negative heavy input (shape {output_negative_heavy.shape}):\n{output_negative_heavy}")
    print(f"Output of first layer (ReLU) after forward pass: {mlp_relu.layers[0].output}")