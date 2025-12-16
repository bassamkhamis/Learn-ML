from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from sklearn.datasets import load_iris
import numpy as np

# Create a Sequential model
model = Sequential()

# Add an input layer with 10 neurons, 4 input features  
model.add(Dense(10, input_shape=(4,), activation='sigmoid'))

# Add a hidden layer with 10 neurons
model.add(Dense(10, activation='sigmoid'))


# Add an output layer with 3 neurons
model.add(Dense(3, activation='softmax'))

# Load dataset
X, y = load_iris(return_X_y=True)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# One-hot encode labels
Y = np.eye(3)[y]

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Summary of the model
model.summary()

# train the model
history = model.fit(
    X, Y,
    epochs=2000,
    batch_size=16,
    verbose=0
)

loss, acc = model.evaluate(X, Y, verbose=0)
print(f"Training accuracy: {acc * 100:.2f}%")
