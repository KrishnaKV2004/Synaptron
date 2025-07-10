# Synaptron Documentation

**Version:** 1.0.0
**Framework Type:** Educational Deep Learning Framework built with NumPy

---

## Table of Contents

1. Introduction
2. Installation
3. Core Components

   * Activations
   * Layers
   * Losses
   * Optimizers
   * Models

4. Usage Guide
5. Advanced Topics
6. Contribution Guide
7. Contact

---

## 1. Introduction

Synaptron is a lightweight, beginner-friendly neural network framework designed to help users understand the internal mechanics of deep learning. It features a modular design where each part of the training pipeline is clearly separated, resembling the structure of popular libraries like Keras.

---

## 2. Installation

```bash
git clone https://github.com/yourusername/Synaptron.git
cd Synaptron
pip install .
```

Synaptron will be globally accessible as a Python package after installation.

---

## 3. Core Components

### 3.1 Activations (`Synaptron.activations`)

Available activations:

* **ReLU**: `from Synaptron.activations.relu import ReLU`
* **Sigmoid**: `from Synaptron.activations.sigmoid import Sigmoid`
* **Linear**: `from Synaptron.activations.linear import Linear`

Each activation implements:

```python
def forward(x):
    # Returns activated output

def backward(x):
    # Returns gradient for backpropagation
```

### 3.2 Layers (`Synaptron.layers`)

Currently supported:

* **Dense Layer**: Fully connected layer

Usage:

```python
from Synaptron.layers import Dense
layer = Dense(input_dim=3, units=4, activation='relu')
```

Methods:

* `forward(inputs)`
* `backward(dvalues)`
* `update_weights(learning_rate)`

### 3.3 Losses (`Synaptron.losses`)

* **Mean Squared Error (MSE)**

Usage:

```python
from Synaptron.losses import mse
loss = mse.mse_loss(y_true, y_pred)
```

### 3.4 Optimizers (`Synaptron.optimizers`)

* **SGD (Stochastic Gradient Descent)**

Usage:

```python
from Synaptron.optimizers import sgd
optimizer = sgd.SGD(learning_rate=0.01)
```

Methods:

* `step(layers)`

### 3.5 Models (`Synaptron.models`)

Currently available:

* **Sequential Model**

Usage:

```python
from Synaptron.models import Sequential
model = Sequential()
```

Methods:

* `add(layer)` – Add layers sequentially
* `fit(X, y, epochs, learning_rate)` – Train the model
* `predict(X)` – Inference

---

## 4. Usage Guide

```python
import numpy as np
from Synaptron.models import Sequential
from Synaptron.layers import Dense

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(input_dim=2, units=4, activation='relu'))
model.add(Dense(input_dim=4, units=1, activation='sigmoid'))

model.fit(X, y, epochs=5000, learning_rate=0.1)
predictions = model.predict(X)
print("Predictions:\n", predictions)
```

---

## 5. Advanced Topics

Planned features:

* Add Dropout, BatchNorm, Conv2D layers
* Implement more loss functions (CrossEntropy)
* Add support for validation and metrics
* Save and load models

---

## 6. Contribution Guide

To contribute:

* Fork the repository
* Create a new branch for your feature
* Submit a pull request with a detailed description

Follow PEP8 style and keep code modular.

---

## 7. Contact

**Author**: Krishna Verma
**GitHub**: [github.com/KrishnaKV2004](https://github.com/KrishnaKV2004)
**Email**: [krishnaverma.0227@gmail.com](mailto:krishnaverma.0227@gmail.com)

---

Thank you for using Synaptron ! If you enjoy the project, consider giving it a ⭐ on GitHub !