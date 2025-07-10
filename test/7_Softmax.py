import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def softmax(x):
    
    # x shape: (n_samples, n_features)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation_fn_name='sigmoid'):
        self.weights = np.random.rand(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.output = None
        self.activation_fn_name = activation_fn_name

        if activation_fn_name == 'sigmoid': self.activation_fn = sigmoid
        elif activation_fn_name == 'relu': self.activation_fn = relu
        elif activation_fn_name == 'softmax': self.activation_fn = softmax
        elif activation_fn_name == 'linear': self.activation_fn = linear
        else: raise ValueError(f"Unsupported activation: {activation_fn_name}")

    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation_fn(weighted_sum)
        return self.output

class MLP:
    def __init__(self): self.layers = []
    def add_layer(self, layer): self.layers.append(layer)
    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers: current_input = layer.forward(current_input)
        return current_input

if __name__ == "__main__":
    
    # ----- MLP for Classification -----
    X_clf = np.array([[1., .5, -1., 2.],
                      [-.5, 0., 1.5, -2.5]])
    print(f"Input X for classification (shape {X_clf.shape}):\n{X_clf}")

    mlp_clf = MLP()
    mlp_clf.add_layer(DenseLayer(4, 8, 'relu'))
    mlp_clf.add_layer(DenseLayer(8, 5, 'relu'))
    mlp_clf.add_layer(DenseLayer(5, 3, 'softmax'))  # 3 classes

    print(f"\nMLP for Classification created: {len(mlp_clf.layers)} layers.")
    for i, layer in enumerate(mlp_clf.layers):
        print(f"  Layer {i+1}: {layer.n_inputs} inputs, {layer.n_neurons} neurons, Activation: {layer.activation_fn_name}")

    out_clf = mlp_clf.forward(X_clf)
    print(f"\nOutput (Softmax):\n{out_clf}")
    print(f"Sum of probabilities per sample: {np.sum(out_clf, axis=1)}")

    # ----- MLP for Regression -----
    X_reg = np.array([[0.1, 0.2, 0.3, 0.4]])
    print(f"\nInput X for regression (shape {X_reg.shape}):\n{X_reg}")

    mlp_reg = MLP()
    mlp_reg.add_layer(DenseLayer(4, 10, 'relu'))
    mlp_reg.add_layer(DenseLayer(10, 1, 'linear'))  # Single regression output

    print(f"\nMLP for Regression created: {len(mlp_reg.layers)} layers.")
    for i, layer in enumerate(mlp_reg.layers):
        print(f"  Layer {i+1}: {layer.n_inputs} inputs, {layer.n_neurons} neurons, Activation: {layer.activation_fn_name}")

    out_reg = mlp_reg.forward(X_reg)
    print(f"\nOutput (Linear):\n{out_reg}")