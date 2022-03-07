import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

#Reading The Data
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

#Seperating the features and the result
X = df.drop(columns='DEATH_EVENT', axis=1)
Y = df["DEATH_EVENT"]

#Splitting the data into a training set and a testing set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
print("X: {} ".format(X))
print("Y: {} ".format(Y))

#Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

#Accuracy on train data
#predicted values of the training set
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on training data ", train_data_accuracy)

#Accuracy on test data
#predicted values of the testing set
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on training data ", test_data_accuracy)

#Building the predictive system
input_data = (75, 0	, 582, 0, 20, 1, 265000, 1.9, 130, 1, 0, 4)
#change input data to numpy array
input_npArray = np.asarray(input_data)


