# logistic linear: the key idea its loss function(logstic loss, log loss, cross entropy loos) 
# in euclidean distance loss, e = (y_hat_i - yi)^2, y_hat range ]-infinity, +infinity[ makes it unfair.
# use sigmoid activation function at the output made the range[0-1], however make the loss function not convex and fail to apply GD algor
# we need another way to formulate the loss function(logstic loss function) 
# j = sum[yi*ln(y_hat_i) + (1-yi)*ln(1-y_hat_i)]
# y_hat_i = sigmoid(xi.t * w)
# d_error = xi*(y_hat_i - yi)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return  1 / (1 + np.exp(-z))
# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Binary classification: class 0 (1) vs others (0)
y_binary = np.where(y == 0, 1, 0)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# *** Scale features ***
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# setup
np.random.seed(42)
W = np.random.randn(5, 1) * 0.01  # Small random weights including bias
learning_rate = 0.1
n_epochs = 200 
n_train = len(X_train)

print(f"Training Perceptron for {n_epochs} epochs...\n")

for epoch in range(1, n_epochs + 1):    
    # Shuffle indices each epoch for better stochastic updates
    indices = np.random.permutation(n_train)
    
    loss = 0   
    for i in indices:
        xi = np.concatenate(([1], X_train[i])).reshape(-1, 1)  # (5, 1)
        y_hat_i = sigmoid(np.dot(xi.T, W)[0, 0])
        true_label = y_train[i]

        # loss
        loss += true_label * np.log(y_hat_i) + (1 - true_label) * np.log(1 - y_hat_i)
        dw = xi * (y_hat_i - true_label)
        W -= learning_rate * dw
        
    if epoch % 20 == 0 or epoch <= 5:
        print(f"Epoch {epoch:3d} | loss: {loss}")
 
print("\nTraining completed.\n")

#=== Test predictions ===
correct = 0
print("Test set wrong predictions:")
for i in range(len(X_test)):
    xi = np.concatenate(([1], X_test[i])).reshape(-1, 1)
    score = sigmoid(np.dot(xi.T, W)[0, 0])
    y_pred = 1 if score > 0.8 else 0
    
    true_class = 0 if y_test[i] == 1 else "non-zero"
    pred_class = 0 if y_pred == 1 else "non-zero"
    
    if y_pred != y_test[i]:
        print(f"Sample {i}: True={true_class}, Predicted={pred_class} → Wrong")
    else:
        correct += 1

accuracy = correct / len(X_test)
print(f"\nTest Accuracy: {correct}/{len(X_test)} = {accuracy:.3f}")

# stochastic gradient descent 
# W_new = W_old +/- learning_rate * Xi
# W [w0, w1, w2, w3, w4], we have dataset of feature lengrh = d = 4