import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)
def linear(x): return x
# Softmax not included for simplicity, as we focus on regression tasks

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation_fn_name='sigmoid',
                 weight_init_strategy='random_scaled', weight_init_scale=0.01):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation_fn_name = activation_fn_name
        self.weight_init_strategy = weight_init_strategy
        self.weight_init_scale = weight_init_scale # Relevant for 'random_scaled'

        if weight_init_strategy == 'random_scaled':
            self.weights = np.random.randn(self.n_inputs, self.n_neurons) * self.weight_init_scale
        elif weight_init_strategy == 'xavier_normal':
            stddev = np.sqrt(2.0 / (self.n_inputs + self.n_neurons))
            self.weights = np.random.randn(self.n_inputs, self.n_neurons) * stddev
        elif weight_init_strategy == 'he_uniform':
            limit = np.sqrt(6.0 / self.n_inputs)
            self.weights = np.random.uniform(-limit, limit, (self.n_inputs, self.n_neurons))
        else:
            raise ValueError(f"Unsupported weight initialization strategy: {weight_init_strategy}")

        self.biases = np.zeros((1, n_neurons)) # Biases usually start at 0
        self.output = None

        if activation_fn_name == 'sigmoid': self.activation_fn = sigmoid
        elif activation_fn_name == 'relu': self.activation_fn = relu
        elif activation_fn_name == 'linear': self.activation_fn = linear
        else: raise ValueError(f"Unsupported activation: {activation_fn_name}")

    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation_fn(weighted_sum)
        return self.output


if __name__ == "__main__":
    X_sample = np.array([[0.2, -0.4, 1.5]])
    print(f"Input X (shape {X_sample.shape}):\n{X_sample}\n")

    # Layer 1: Random scaled initialization (default strategy, custom scale)
    l1 = DenseLayer(3, 64, 'relu', weight_init_strategy='random_scaled', weight_init_scale=0.1)
    # Layer 2: Xavier/Glorot normal initialization
    l2 = DenseLayer(64, 32, 'sigmoid', weight_init_strategy='xavier_normal')
    # Layer 3: He uniform initialization
    l3 = DenseLayer(32, 2, 'relu', weight_init_strategy='he_uniform')
    # Layer 4: Default random scaled initialization (scale=0.01)
    l4 = DenseLayer(2, 1, 'linear')


    print(f"L1 ({l1.n_inputs}i, {l1.n_neurons}n, Act:{l1.activation_fn_name}, Init:{l1.weight_init_strategy} (scale={l1.weight_init_scale}))")
    print(f"  Weights sample:\n{l1.weights[:2,:2]}\n")

    print(f"L2 ({l2.n_inputs}i, {l2.n_neurons}n, Act:{l2.activation_fn_name}, Init:{l2.weight_init_strategy})")
    print(f"  Weights sample:\n{l2.weights[:2,:2]}\n")

    print(f"L3 ({l3.n_inputs}i, {l3.n_neurons}n, Act:{l3.activation_fn_name}, Init:{l3.weight_init_strategy})")
    print(f"  Weights sample:\n{l3.weights[:2,:2]}\n")

    print(f"L4 ({l4.n_inputs}i, {l4.n_neurons}n, Act:{l4.activation_fn_name}, Init:{l4.weight_init_strategy} (scale={l4.weight_init_scale}))")
    print(f"  Weights sample:\n{l4.weights[:2,:2]}\n")

    # Expected standard deviations (approximate):
    # L1 (random_scaled, scale=0.1): stddev should be close to 0.1
    # L2 (xavier_normal, 64 inputs, 32 neurons): stddev = sqrt(2 / (64 + 32)) = sqrt(2/96) approx 0.144
    # L3 (he_uniform, 32 inputs): For U(-limit,limit), stddev = limit/sqrt(3). limit = sqrt(6/32). So stddev = sqrt(6/32)/sqrt(3) = sqrt(2/32) = sqrt(1/16) = 0.25
    # L4 (random_scaled, scale=0.01): stddev should be close to 0.01
    print("\nApproximate expected standard deviations for initialized weights:")
    print(f"  L1 (random_scaled, scale={l1.weight_init_scale}): Expected std ~ {l1.weight_init_scale:.4f}, Actual std: {np.std(l1.weights):.4f}")
    xavier_expected_std_l2 = np.sqrt(2.0 / (l2.n_inputs + l2.n_neurons))
    print(f"  L2 (xavier_normal): Expected std ~ {xavier_expected_std_l2:.4f}, Actual std: {np.std(l2.weights):.4f}")
    he_expected_std_l3 = np.sqrt(2.0 / l3.n_inputs) # stddev for He uniform is sqrt(2/fan_in)
    print(f"  L3 (he_uniform): Expected std ~ {he_expected_std_l3:.4f}, Actual std: {np.std(l3.weights):.4f}")
    print(f"  L4 (random_scaled, scale={l4.weight_init_scale}): Expected std ~ {l4.weight_init_scale:.4f}, Actual std: {np.std(l4.weights):.4f}")