from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X, y = iris.data, iris.target

print("X", X)
scaler = StandardScaler()   # Sigma and Mean method
X = scaler.fit_transform(X)

# minMax = MinMaxScaler()
# X= minMax.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("y_test", y_test)
# print("X_test", X_test)

k_values = [3, 5, 7]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    # print("y_pre", y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for K={k}: {accuracy:.2f}")