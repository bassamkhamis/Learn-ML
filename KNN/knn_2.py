import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Load the Car Evaluation dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(url, header=None, names=columns)

# print("df", df)

# Features and target
X = df.drop('class', axis=1)  # All categorical features
y = df['class']

# Define ordered categories for each feature
buying_cats = ['low', 'med', 'high', 'vhigh']
maint_cats = ['low', 'med', 'high', 'vhigh']
doors_cats = ['2', '3', '4', '5more']
persons_cats = ['2', '4', 'more']
lug_boot_cats = ['small', 'med', 'big']
safety_cats = ['low', 'med', 'high']

# Preprocessor with OrdinalEncoder
categorical_features = X.columns.tolist()  # All are categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=[buying_cats, maint_cats, doors_cats, persons_cats, lug_boot_cats, safety_cats]), categorical_features)
    ])

# Preprocessing: One-hot encode all categorical features
# categorical_features = X.columns.tolist()  # All are categorical
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ])

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# print("X_test", X_test)

# Test K-NN with different K values
k_values = [3, 5, 7, 15, 25, 51]
for k in k_values:
    # Test with Hamming(way to compute the distance like euclideans)
    knn_hamming = KNeighborsClassifier(n_neighbors=k, metric='hamming')
    knn_hamming.fit(X_train, y_train)
    y_pred_hamming = knn_hamming.predict(X_test)
    acc_hamming = accuracy_score(y_test, y_pred_hamming)
    print(f"Accuracy for K={k} with Hamming: {acc_hamming:.2f}")

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for K={k} with manhattan: {accuracy:.2f}")

# The best K is the one with the highest accuracy (run to see results)