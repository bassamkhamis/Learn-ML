from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from qpsolvers import solve_qp


class QPSVM:
    def __init__(self, C=None, kernel="linear", degree=3, gamma=2):
        """
        C = None → hard margin SVM
        C = float → soft margin SVM
        """
        self.C = C
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None

        self.X_train = None
        self.y_train = None

        self.kernel_type = kernel
        self.degree = degree
        self.gamma = gamma

    def _kernel(self, X1, X2):
        if self.kernel_type == "linear":
            return X1 @ X2.T
        elif self.kernel_type == "poly":
            return (1 +  X1 @ X2.T) ** self.degree
        elif self.kernel_type == "rbf":
            # ||x - x'||² = (x² + x'² - 2x·x')
            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
            return np.exp(-self.gamma * (X1_sq + X2_sq - 2 * X1 @ X2.T))
        else:
            raise ValueError("Unknown kernel")

    def fit(self, X, y):
        """
        Train SVM using QP dual.
        X: (N, d)
        y: (N,) with labels in {-1, +1}
        """
        N, d = X.shape
        y = y.astype(float)

        self.X_train = X
        self.y_train = y

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
        A = y.reshape(1, -1).astype(float)
        b = np.array([0.0])

        # --- Solve QP -----------------------------
        alpha = solve_qp(P, q, G, h, A, b, solver="cvxopt")

        alpha = np.array(alpha)
        self.alpha = alpha

        # Support vectors: α_i > 1e-6
        sv = (alpha > 1e-6)
        self.support_vectors = sv
        self.alpha_sv = alpha[sv]
        self.X_sv = X[sv]
        self.y_sv = y[sv]

        # Compute w (only for linear kernel)
        if self.kernel_type == "linear":
            self.w = np.sum((self.alpha_sv * self.y_sv)[:, None] * self.X_sv, axis=0)
        else:
            self.w = None  # w not computable in original space for non-linear kernels

        # Compute b using margin support vectors
        if self.C is None:  
            margin_sv = sv  # All support vectors for hard margin
        else: # 0 < alpha < c
            margin_sv = (alpha > 1e-6) & (alpha < self.C - 1e-6)  # Free support vectors for soft margin

        if np.sum(margin_sv) == 0:
            # Fallback to all support vectors if no free SVs
            margin_sv = sv

        X_margin = X[margin_sv]
        y_margin = y[margin_sv]

        # Compute sum alpha_j y_j K(x_j, x_i) for each margin i
        K_margin_sv = self._kernel(self.X_sv, X_margin)
        sum_terms = (self.alpha_sv * self.y_sv) @ K_margin_sv

        # y
        b_values = y_margin - sum_terms
        self.b = np.mean(b_values)    

        # # Compute w (only linear SVM) w=i=1∑​αi​yi​xi
        # if self.kernel_type == "linear":
        #     self.w = np.sum((alpha * y)[:, None] * X, axis=0)
        #     # Compute b using KKT conditions (average over support vectors)
        #     sv_indices = np.where(sv)[0]
        #     bs = []
        #     for i in sv_indices:
        #         bs.append(y[i] - np.dot(self.w, X[i]))
        #     self.b = np.mean(bs)

    def decision_function(self, X):
        """Compute w·x + b"""
        if self.kernel_type == "linear":
            return X @ self.w + self.b
        
        K = self._kernel(self.X_sv, X)   # shape (n_sv, n_test)
        return (self.alpha_sv * self.y_sv) @ K + self.b

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

def plot_decision_boundary_2(model, X, y):
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plotting requires 2D data")

    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0], colors="k", linewidths=2)

    plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", edgecolors="k")
    plt.title(f"SVM Decision Boundary ({model.kernel_type})")
    plt.show()


X, y = make_blobs(n_samples=150, centers=2, n_features=2, random_state=0, cluster_std=0.60)
y = 2 * (y - 1/2)   # Convert {0,1} → {-1,+1}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

model = QPSVM(C=1.0)
model.fit(X_train, y_train)

print("Weights:", model.w)
print("Bias:", model.b)
print("Accuracy:", model.accuracy(X_test, y_test))
plot_decision_boundary_2(model, X_train, y_train)


X2, y2 = noisy_circles = make_circles(n_samples=200, factor=.5, noise=.05)
y2 = np.where(y2 <= 0, -1, 1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, shuffle=True, random_state=1)

model = QPSVM(C=3, kernel="rbf", degree=4)
model.fit(X2_train, y2_train)
print("Bias:", model.b)
print("Accuracy:", model.accuracy(X2_test, y2_test))
plot_decision_boundary_2(model, X2_train, y2_train)

X3, y3 = make_moons(n_samples=200, noise=.05)
y3 = np.where(y3 <= 0, -1, 1)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, shuffle=True, random_state=1)

model = QPSVM(C=3, kernel="rbf")
model.fit(X3_train, y3_train)
print("Bias:", model.b)
print("Accuracy:", model.accuracy(X3_test, y3_test))
plot_decision_boundary_2(model, X3_train, y3_train)
