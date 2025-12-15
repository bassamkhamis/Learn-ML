# simple ANN 
# dataset: IRIS

import numpy as np
from sklearn.datasets import load_iris

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    # a is sigmoid(z)
    return a * (1 - a)

def mse(y_hat, y):
    return np.mean((y_hat - y) ** 2)


def mse_derivative(y_hat, y):
    return 2 * (y_hat - y) / y.shape[0]


np.random.seed(42)

D = 4   # input size
H = 10   # hidden neurons
C = 3   # output size

W1 = np.random.randn(D, H) * np.sqrt(1 / D)
b1 = np.zeros((1, H))

W2 = np.random.randn(H, C) * np.sqrt(1 / H)
b2 = np.zeros((1, C))

def forward(X):
    # Hidden layer
    z1 = X @ W1 + b1          # (N, H), net_1
    a1 = sigmoid(z1)          # (N, H), output_1

    # Output layer
    z2 = a1 @ W2 + b2         # (N, C), net_2
    y_hat = sigmoid(z2)       # (N, C), output_2

    return z1, a1, z2, y_hat


def backward(X, y, z1, a1, z2, y_hat, lr):
    global W1, b1, W2, b2

    N = X.shape[0]

    # dE/dy_hat
    d_y_hat = 2 * (y_hat - y) / N

    d_z2 = d_y_hat * sigmoid_derivative(y_hat) # d_E/d_output * d_out/d_net = (y_hat - y) * y_hat*(1-yhat), d_out/d_net = Qj(1-Oj), Oj = y_hat

    dW2 = a1.T @ d_z2
    db2 = np.sum(d_z2, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2

    d_a1 = d_z2 @ W2.T # sigma_k * Wkj
    d_z1 = d_a1 * sigmoid_derivative(a1) # sigma_j

    dW1 = X.T @ d_z1
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    # W2 -= lr * dW2
    # b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

# Load dataset
X, y = load_iris(return_X_y=True)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# One-hot encode labels
Y = np.eye(3)[y]

lr = 0.09

for epoch in range(20000):
    z1, a1, z2, y_hat = forward(X)
    loss = mse(y_hat, Y)
    backward(X, Y, z1, a1, z2, y_hat, lr)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss:.4f}")

print("\nPredictions:")
out = forward(X)[-1]
np.round(out, out=out)
accuracy = np.mean(out == Y)
print("Accuracy: ", accuracy)







