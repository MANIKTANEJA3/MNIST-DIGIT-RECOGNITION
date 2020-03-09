#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#read train and test files
train_file = pd.read_csv('mnist_train.csv')
test_file = pd.read_csv('mnist_test.csv')

#first few rows of the test and train files
train_file.head()
test_file.head()


#list of all digits that are going to be predicted
np.sort(train_file.label.unique())

#define the number of samples for training set and for validation set from the training set
num_train,num_validation = int(len(train_file)*0.8),int(len(train_file)*0.2)

#calculating the number of training and validation sets
num_train,num_validation

#generate training data from train_file
x_train,y_train=train_file.iloc[:num_train,1:].values,train_file.iloc[:num_train,0].values

#generate validationn data from train_file
x_validation,y_validation=train_file.iloc[num_train:,1:].values,train_file.iloc[num_train:,0].values


#prints the training number and the pixel*pixel figure  for the features and their corresponding labels
print(x_train.shape)
print(y_train.shape)
print(x_validation.shape)
print(y_validation.shape)


#visualising the data shows an image of 1
index=3
print("Label: " + str(y_train[index]))
plt.imshow(x_train[index].reshape((28,28)),cmap='gray')
plt.show()


#fit a Random Forest classifier
clf=RandomForestClassifier()
clf.fit(x_train,y_train)


#predict value of label using classifier
prediction_validation = clf.predict(x_validation)


print("Validation Accuracy: " + str(accuracy_score(y_validation,prediction_validation)))


print("Validation Confusion Matrix: \n" + str(confusion_matrix(y_validation,prediction_validation)))


index=3
print("Predicted " + str(y_validation[y_validation!=prediction_validation][index]) + " as " + 
     str(prediction_validation[y_validation!=prediction_validation][index]))
plt.imshow(x_validation[y_validation!=prediction_validation][index].reshape((28,28)),cmap='gray')
#predict test data

#generate training data from train_file

num_test = int(len(test_file)*1.0)
x_test,y_test=test_file.iloc[:num_test,1:].values,test_file.iloc[:num_test,0].values
print(x_test.shape)
print(y_test.shape)
prediction_test = clf.predict(x_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction_test))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction_test)))
print(accuracy_score(y_test, prediction_test))