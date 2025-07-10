import numpy as np
from models import Sequential
from layers import Dense

# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Build the model
print("\nBuilding Synaptron Sequential Model...")
model = Sequential()
model.add(Dense(input_dim=2, units=4, activation='relu'))
model.add(Dense(input_dim=4, units=1, activation='sigmoid'))

# Train the model
print("\nTraining on XOR dataset...")
model.fit(X, y, epochs=5000, learning_rate=0.1, verbose=False)

# Predict
print("\nTesting predictions on XOR inputs:")
predictions = model.predict(X)

for i, (inp, pred) in enumerate(zip(X, predictions)):
    print(f"Input {i+1}: {inp} => Predicted: {pred[0]:.4f} | Expected: {y[i][0]}")

# Optional: print in rounded matrix form
print("\nRounded Predictions Matrix:")
print(np.round(predictions, 3))