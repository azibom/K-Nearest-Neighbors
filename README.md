# K-Nearest-Neighbors
I try to implement the KNN

### first you need to load your datas and then it is nice to make chart with your datas :art:
```python
# import requirement
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import datasets
import pandas as pd
# load data
iris = datasets.load_iris() 
# make them useable
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target
# show the relation in one chart
pd.plotting.scatter_matrix(iris_data, c=iris.target)
plt.show()
```
### now i implement the knn and i expect it to  predict for me but only there is one problem i don't know the prediction is true of not
```python
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
```
### this time we want to divide our data to the two groupe and train_data and test_data so this time we can examine our model :fire:
```python
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

# show the accuracy in an easy way
print(knn.score(x_test, y_test)*100)
```
and now you can import data, predict them and even test them

but in the end, it is nice to know two definition 

#### overfitting: when your model gives 100% score to learn data and give a low score in test data
#### under fitting: when your model gives a low score in learn data and can't predict well

I hope this data will be useful to you.



