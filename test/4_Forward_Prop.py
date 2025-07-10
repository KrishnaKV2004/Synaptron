import numpy as np


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))
        self.output = None # To store the output of the forward pass

    def forward(self, inputs):
        """
        Perform a forward pass through the dense layer.
        inputs: A NumPy array of input data. Shape: (n_samples, n_inputs).
        """
        # Calculate weighted sum: inputs (dot) weights
        # inputs shape: (n_samples, n_inputs)
        # self.weights shape: (n_inputs, n_neurons)
        # weighted_sum shape: (n_samples, n_neurons)
        weighted_sum = np.dot(inputs, self.weights)
        
        # TODO: Add biases to the weighted sum. Use NumPy broadcasting to add
        # self.biases (shape: 1, n_neurons) to weighted_sum (shape: n_samples, n_neurons)
        output_before_activation = weighted_sum + self.biases 
        
        # Apply activation function
        self.output = sigmoid(output_before_activation)
        return self.output


if __name__ == "__main__":
    # Initialize a layer: 3 input features, 2 neurons
    n_inputs = 3
    n_neurons = 2
    layer = DenseLayer(n_inputs, n_neurons)
    
    print(f"Layer weights (shape {layer.weights.shape}):\n{layer.weights}")
    print(f"Layer biases (shape {layer.biases.shape}):\n{layer.biases}")

    # Sample input data: 2 samples, each with 3 features
    # This represents a batch of 2 data points.
    sample_batch_inputs = np.array([
        [1.0, 2.0, 3.0],  # Sample 1
        [0.5, 1.5, 2.5]   # Sample 2
    ])
    print(f"\nInput batch data (shape {sample_batch_inputs.shape}):\n{sample_batch_inputs}")

    # Perform forward pass
    layer_output = layer.forward(sample_batch_inputs)
    
    print(f"\nOutput of the layer after forward pass (shape {layer_output.shape}):\n{layer_output}")