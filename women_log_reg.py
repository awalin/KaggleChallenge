from sklearn import linear_model, neighbors
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle
import numpy as np
# fix random seed for reproducibility
numpy.random.seed(0)
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split

# import some data to play with

name = 'train'
file = name + '_big_4.csv'
df = pd.read_csv(file)
print(name + '  file read ', df.shape)
df.fillna('NA', inplace=True)

Y = df['is_female']
df = df.drop('is_female', axis=1)

X = preprocessing.normalize(df.values)


model = svm.SVC()

# model.fit(X, y)
# # model = linear_model.LogisticRegression(C=1e5)
# model = neighbors.KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15,
													random_state=0)
model.fit(X_train, y_train)

scores = model.score(X_test, y_test)
print scores

