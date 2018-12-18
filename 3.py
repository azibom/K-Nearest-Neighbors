# import requirement
from datetime import datetime
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# load data
iris = datasets.load_iris() 
# make them useable
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

x = iris.data
y = iris.target

# test the model 
# make test and train it
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

# k-nierest neighbor
knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
knn.fit(x_train, y_train)

# calculate the prediction
y_predict = knn.predict(x_test)

# show the accurracy in easy way
print(knn.score(x_test, y_test)*100)
