from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

class Knn:
    k = 3
    def __init__(self, k):
        self.k = k

    def calculate_square_distance(self, sample_1, sample_2):
        assert(len(sample_1) == len(sample_2))    
        dis = 0
        for i in range(0, len(sample_1)):
            dis += (sample_1[i] - sample_2[i]) ** 2
        return dis

    def fit(self, x, y):
        self.y = y

        self.mean = x.mean(axis = 0)
        self.std = x.std(axis = 0)
        self.x = (x - self.mean) / self.std

    def _normalize_sample(self, sample):
        return (sample - self.mean) / self.std   

    def _predict(self, sample):
        sample = self._normalize_sample(sample)
        distances = []
        for i in range(0, len(self.x)):
            dis = self.calculate_square_distance(self.x[i], sample)
            distances.append([dis, self.y[i]])       

        distances.sort(key=lambda pair: pair[0])
        # Takes the first k and performs majority voting
        k_labels = [label for(_, label) in distances[:self.k]]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, samples):
        res = np.zeros(len(samples), dtype=int)
        for i in range(0, len(samples)):
            res[i] = self._predict(samples[i])
        return res 

    def accurcy(self, d1, d2):
        assert(len(d1) == len(d2))
        incorrect = 0
        for i in range(0, len(d1)):
            if d1[i] != d2[i]:
                incorrect += 1
        return (len(d1) - incorrect) / len(d1)        

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Ks = [3, 5, 7, 9, 13, 15]

for k in Ks:
    knn = Knn(k)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    print(f'accurcy {knn.accurcy(y_predict, y_test)}')
            