from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from qpsolvers import solve_qp


class QPSVM:
    def __init__(self, C=None):
        """
        C = None → hard margin SVM
        C = float → soft margin SVM
        """
        self.C = C
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None

    def _kernel(self, X1, X2):
        """Linear kernel"""
        return X1 @ X2.T

    def fit(self, X, y):
        """
        Train SVM using QP dual.
        X: (N, d)
        y: (N,) with labels in {-1, +1}
        """
        N, d = X.shape
        y = y.astype(float)

        # Kernel matrix
        K = self._kernel(X, X)

        # --- QP Matrices -------------------------------------------------

        # P_{ij} = y_i y_j K(x_i, x_j)
        P = np.outer(y, y) * K
        # 🔥 Fix: ensure positive definiteness
        P += 1e-6 * np.eye(N)

        # q_i = -1
        q = -np.ones(N)

        # Inequality constraints G α ≤ h
        if self.C is None:
            # Hard margin: α ≥ 0  →  -I α ≤ 0
            G = -np.eye(N)
            h = np.zeros(N)
        else:
            # Soft margin: 0 ≤ α ≤ C
            G = np.vstack([
                -np.eye(N),
                np.eye(N)
            ])
            h = np.hstack([
                np.zeros(N),
                self.C * np.ones(N)
            ])

        # Equality constraint ∑ y_i α_i = 0
        A = y.reshape(1, -1)
        b = np.array([0.0])

        # --- Solve QP -----------------------------
        alpha = solve_qp(P, q, G, h, A, b, solver="quadprog")

        alpha = np.array(alpha)
        self.alpha = alpha

        # Support vectors: α_i > 1e-4
        sv = alpha > 1e-5
        self.support_vectors = sv

        # Compute w (only linear SVM)
        self.w = np.sum((alpha * y)[:, None] * X, axis=0)

        # Compute b using KKT conditions (average over support vectors)
        sv_indices = np.where(sv)[0]
        bs = []
        for i in sv_indices:
            bs.append(y[i] - np.dot(self.w, X[i]))
        self.b = np.mean(bs)

        return self

    def decision_function(self, X):
        """Compute w·x + b"""
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(8, 6))

    # Scatter plot of data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=40)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    # Decision function on grid
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[0], colors='k')      # decision boundary
    plt.contour(xx, yy, Z, levels=[-1], colors='k', linestyles='dashed')  # -1
    plt.contour(xx, yy, Z, levels=[1], colors='k', linestyles='dashed')   # +1

    # Highlight support vectors
    sv = np.where(model.support_vectors)[0]
    plt.scatter(X[sv, 0], X[sv, 1], s=120,
                facecolors='none', edgecolors='k', linewidths=1.5)

    plt.title("SVM Decision Boundary")
    plt.show()


X, y = make_blobs(n_samples=150, centers=2, n_features=2, random_state=0, cluster_std=0.60)
y = 2 * (y - 1/2)   # Convert {0,1} → {-1,+1}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

model = QPSVM(C=1.0)
model.fit(X_train, y_train)

print("Weights:", model.w)
print("Bias:", model.b)
print("Accuracy:", model.accuracy(X_test, y_test))
plot_decision_boundary(model, X_train, y_train)