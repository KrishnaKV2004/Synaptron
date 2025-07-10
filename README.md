# 🚀 Synaptron

**Synaptron** is a minimal, modular, and educational deep learning framework built from scratch in Python using **NumPy**.
Inspired by **Keras**, Synaptron lets you create, train, and evaluate neural networks easily while understanding the underlying mechanics.

---

## ✨ Features (v1.0)

* 🧱 **Modular layers**: Dense layer with forward and backward propagation
* ⚡️ **Activations**: ReLU, Sigmoid, Linear
* 🧾 **Loss function**: Mean Squared Error (MSE)
* 🏋️ **Optimizer**: Stochastic Gradient Descent (SGD)
* 🧪 Easy-to-follow codebase designed for learning and customization
* 🔄 Fully supports training loops with gradient updates
* 🐍 Pure Python + NumPy, no heavy dependencies

---

## 📦 Installation

Clone the repo and install dependencies locally :

```bash
git clone https://github.com/KrishnaKV2004/Synaptron.git
cd Synaptron
pip install .
```

*This installs Synaptron and its dependency (`numpy`) on your system, so you can import it anywhere!*

---

## ⚙️ Usage Example

Here’s how you can train a simple XOR model with Synaptron:

```python
import numpy as np
from Synaptron.models import Sequential
from Synaptron.layers import Dense

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Build model
model = Sequential()
model.add(Dense(input_dim=2, units=4, activation='relu'))
model.add(Dense(input_dim=4, units=1, activation='sigmoid'))

# Train model
model.fit(X, y, epochs=5000, learning_rate=0.1)

# Predict
predictions = model.predict(X)
print("Predictions:\n", predictions)
```

---

## 🖥️ Project Structure

```
Synaptron/
├── 📁 activations/
│   ├── __init__.py
│   ├── relu.py
│   ├── sigmoid.py
│   └── linear.py
├── 📁 layers/
│   ├── __init__.py
│   └── dense.py
├── 📁 losses/
│   ├── __init__.py
│   └── mse.py
├── 📁 models/
│   ├── __init__.py
│   └── sequential.py
├── 📁 optimizers/
│   ├── __init__.py
│   └── sgd.py
├── 📁 utils/
│   ├── __init__.py
│   └── helpers.py
├── __init__.py
├── setup.py
├── main.py
└── README.md
```

---

## 🎯 Goals & Future Plans

* ➕ Add more layer types (Conv2D, Dropout, BatchNorm, etc.)
* 📈 Support additional loss functions (Cross-Entropy, Hinge loss, etc.)
* 🚀 More optimizers (Adam, RMSProp, etc.)
* 📊 Better metrics & evaluation tools
* 🎨 User-friendly APIs inspired by Keras/TensorFlow
* 🔧 Example projects and tutorials for easier learning

---

## 🤝 Contribution

Contributions are very welcome ! Feel free to :

* Fork the repo
* Submit issues and feature requests
* Create pull requests to add new layers, losses, or improvements

---

## 📞 Contact

**Krishna Verma** — [GitHub](https://github.com/KrishnaKV2004) — [krishnaverma.0227@gmail.com](mailto:krishnaverma.0227@gmail.com)

---

Thank you for checking out Synaptron !
Happy coding and happy learning ! 🎉

---

*Made with ❤️ by Krishna Verma*