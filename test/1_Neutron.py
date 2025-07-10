import numpy as np

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
        output = weighted_sum + self.bias

        return output

if __name__ == "__main__" :

    n_inputs = 5
    neuron = Neuron(n_inputs)

    sample_inputs_1 = np.array([1.0, 2.0, 3.0])
    print(f"\nInput to neuron: {sample_inputs_1}")

    output_1 = neuron.forward(sample_inputs_1)
    print(f"Neuron's raw output (weighted sum + bias): {output_1:.4f}")