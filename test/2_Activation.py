import numpy as np


''' Sigmoid activation function '''

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))


class Neuron :

    ''' A simple Neuron class that initializes weights and bias '''

    def __init__(self, n_inputs):
        
        self.weights = np.random.rand(n_inputs) * 0.01
        self.bias = 0.0
        
        print(f"Neuron initialized with {n_inputs} inputs.")
        print(f"Initial weights: {self.weights}")
        print(f"Initial bias: {self.bias}")
    

    ''' Forward pass through the neuron '''

    def forward(self, inputs) :
        
        if inputs.shape != self.weights.shape :
            raise ValueError(f"Input shape {inputs.shape} does not match weights shape {self.weights.shape}.")

        weighted_sum = np.dot(inputs, self.weights)

        raw_output = weighted_sum + self.bias
        activated_output = sigmoid(raw_output)

        return activated_output, raw_output
    

if __name__ == "__main__" :

    num_input_features = 3
    neuron = Neuron(num_input_features)
    print(f"Neuron weights: {neuron.weights}, Bias: {neuron.bias}")

    sample_inputs = np.array([1.0, 2.0, 3.0])
    print(f"\nInput to neuron: {sample_inputs}")

    activated_output, raw_output = neuron.forward(sample_inputs)
    print(f"Neuron's raw output (weighted sum + bias): {raw_output:.4f}")
    print(f"Neuron's activated output (Sigmoid): {activated_output:.4f}")