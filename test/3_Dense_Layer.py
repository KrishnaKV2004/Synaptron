import numpy as np


''' A Dense Layer class that initializes weights and biases for a neural network layer '''

class DenseLayer :

    def __init__(self, n_inputs, n_neurons):
        
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons


if __name__ == "__main__" :

    layer1 = DenseLayer(5, 2)
    print(f"\nLayer 1: {layer1.n_inputs} inputs, {layer1.n_neurons} neurons")
    print(f"Weights shape: {layer1.weights.shape}")
    print(f"Weights (first 2 rows):\n{layer1.weights[:2, :]}")
    print(f"Biases shape: {layer1.biases.shape}")
    print(f"Biases:\n{layer1.biases}")

    layer2 = DenseLayer(3, 4)
    print(f"\nLayer 2: {layer2.n_inputs} inputs, {layer2.n_neurons} neurons")
    print(f"Weights shape: {layer2.weights.shape}")
    print(f"Weights (first 2 rows):\n{layer2.weights[:2, :]}")
    print(f"Biases shape: {layer2.biases.shape}")
    print(f"Biases:\n{layer2.biases}")

    layer3 = DenseLayer(6, 1)
    print(f"\nLayer 3: {layer3.n_inputs} inputs, {layer3.n_neurons} neurons")
    print(f"Weights shape: {layer3.weights.shape}")
    print(f"Weights (first 2 rows):\n{layer3.weights[:2, :]}")
    print(f"Biases shape: {layer3.biases.shape}")
    print(f"Biases:\n{layer3.biases}")