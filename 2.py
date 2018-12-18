# import requirement
from datetime import datetime
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
# load data
iris = datasets.load_iris() 
# make them useable
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target

x = iris.data
y = iris.target
# k-nierest neighbor
knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
knn.fit(x, y)
# now you expect the model to predict for you
sample = np.array([[5, 3, 1, 4]])
print(knn.predict(sample))