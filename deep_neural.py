# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:21:31 2019

@author: Aditya Saxena
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


dataset = pd.read_csv('Lithium-ion-battery-dataset.csv')
X = dataset.iloc[:, [6,7,8,9,13,14]].values
y = dataset.iloc[:,5].values
X = np.nan_to_num(X)

#Enabeling_autoencoders

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Normalizing_the_data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


classifier = Sequential()

#NN_Layer1

classifier.add(Dense(activation = "relu", input_dim = 10, units =6, kernel_initializer = "uniform"))

#NN_Layer2

classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#NN_Layer3

classifier.add(Dense(activation="relu", units=6, kernel_initializer = "uniform"))

#NN_Layer4

classifier.add(Dense(activation = "softmax", units =1,kernel_initializer = "uniform"))

#NN_Compiling

classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

#Training_the_dnn

classifier.fit(X_train, y_train, epochs = 100,batch_size = 10)

#Predict

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Confusion_matrix

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier ,X = X_train, y = y_train, cv=10)
#accuracies.mean()
#accuracies.std()

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop= X_set[:,0].max() + 1, step= 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop= X_set[:,1].max() + 1, step= 0.01))
#plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
 #            alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set ==j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i), label =j)
plt.title('Deep Neural Network(Test)')
plt.xlabel('Cycle Number')
plt.ylabel('Cycle-Prediction')
plt.legend()
plt.show()  

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop= X_set[:,0].max() + 1, step= 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop= X_set[:,1].max() + 1, step= 0.01))
#plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
 #            alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set ==j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i), label =j)
plt.title('Deep Neural Network(Train)')
plt.xlabel('Cycle Number')
plt.ylabel('Cycle-Prediction')
plt.legend()
plt.show()  


