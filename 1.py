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