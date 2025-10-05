import numpy as np
from cvxopt import matrix, solvers
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# need different python version

class SVM:
    def __init__(self):
        self.w = None
        self.b = None
        self.support_vectors = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Gram matrix
        K = np.dot(X, X.T)
        
        # Quadratic programming parameters
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        G = matrix(-np.eye(n_samples))
        h = matrix(np.zeros(n_samples))
        A = matrix(y.astype(float), (1, n_samples))
        b = matrix(0.0)
        
        # Solve QP problem
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(sol['x'])
        
        # Support vectors have non zero alphas
        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.support_vectors = X[sv]
        self.support_y = y[sv]
        
        # Compute w
        self.w = np.sum(self.alphas[:, None] * self.support_y[:, None] * self.support_vectors, axis=0)
        
        # Compute b
        self.b = np.mean(self.support_y - np.dot(self.support_vectors, self.w))
    
    def project(self, X):
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        return np.sign(self.project(X))


# Generate 2D data (2 classes)
X, y = make_blobs(n_samples=50, centers=2, random_state=6)
y = np.where(y == 0, -1, 1)  # labels must be -1, +1

# Train custom SVM
svm = SVM()
svm.fit(X, y)

# Plot decision boundary
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
ax = plt.gca()
xlim = ax.get_xlim()
xx = np.linspace(xlim[0], xlim[1], 50)
yy = -(svm.w[0] * xx + svm.b) / svm.w[1]
plt.plot(xx, yy, 'k-')  # decision boundary
plt.show()
