import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd

with open('data/dataframe.pickle', 'rb') as handle:
	df = pickle.load(handle)

data = df.copy()
data.drop('alphabet', axis=1)
target = df.alphabet
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4,random_state=109) # 70% training and 30% test

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

with open('data/classifier_SVM.pickle', 'wb') as handle:
		pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)