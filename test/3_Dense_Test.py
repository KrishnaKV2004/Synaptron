import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize a dense layer.
        n_inputs: Number of input features (from previous layer or input data).
        n_neurons: Number of neurons in this layer.
        """
        # Weights matrix: shape (n_inputs, n_neurons)
        self.weights = np.random.rand(n_inputs, n_neurons) * 0.1
        # Biases vector: shape (1, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

def count_parameters(n_inputs, n_neurons):
    """
    Compute and return the total number of parameters (weights + biases).
    """
    num_weights = n_inputs * n_neurons
    num_biases = n_neurons
    return num_weights + num_biases

if __name__ == "__main__":
    test_cases = [
        (4, 3),
        (5, 2),
        (3, 4),
        (6, 1),
    ]
    for n_inputs, n_neurons in test_cases:
        total = count_parameters(n_inputs, n_neurons)
        print(f"DenseLayer with {n_inputs} inputs and {n_neurons} neurons:")
        print(f"  Total parameters: {total}")