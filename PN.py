#!/usr/bin/env python
# coding: utf-8

# ### Load the dependencies

# import xlsxwriter
import pylightxl as xl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle

import pn_config

from sklearn.preprocessing import MinMaxScaler

import logging

# logging.basicConfig(level = log.info)

logging.basicConfig(filename=f"{pn_config.log_file}", filemode='w', format=f'{pn_config.log_format}', level = logging.DEBUG)
log = logging.getLogger("Predict_Now")

log.warning('Watch out!')  # will print a message to the console
log.info('I told you so')  # will not print anything


### Load the dataset


# ### Read the data file

# readxl returns a pylightxl database that holds all worksheets and its data
log.info(f"Load file: {pn_config.project_path}/{pn_config.excel_file}")
db = xl.readxl(fn=f'{pn_config.project_path}/{pn_config.excel_file}')

# ### Load the rows into a list
log.info(f"Read data from sheet: {pn_config.sheet_name}")
file_rows = []

for row in db.ws(ws=f'{pn_config.sheet_name}').rows:
    file_rows.append(row)


# ### Load the rows into a pandas dataframe
log.info("Create pandas dataframe")
df = pd.DataFrame(file_rows[1:])
df.columns = file_rows[0]
df.head()

# Assume that the data is balanced

# ### Create the features database, X
log.info("Create features dataframe")
X = df.drop(columns=["Time","Returns"])

# ### Create the target column, Y
log.info("Create target list")
y = df["Returns"]

# ### Split the dataset into training and testing
log.info("Create train/test split")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05)


# ### Create variables that will hold the information for the best performing model
log.info("Create global variables for the best-performing model")
global max_score, max_model_name, max_model
max_score = 0
max_model = None
max_model_name = ""


# ### Create a function that implements the following steps:
# * that fits a given model to the data; both passed in as arguments 
# * test the training and testing performances of the model
# * display the confusion matrix for actual vs predicted for the test dataset
# * if the test performance is better than the current best performance, update the information for best performing model

def train_test(model, model_name, X_train, X_test, y_train, y_test):
    
	global max_score, max_model_name, max_model 

	log.info(f"Training {model_name}")
	model.fit(X_train, y_train)
	
	log.info("Predicting text data")
	y_pred = model.predict(X_test)
    
	log.info("Generate performance scores")
	test_score = model.score(X_test, y_test)
	
	log.info("score on test: " + str(test_score))
	log.info("score on train: "+ str(model.score(X_train, y_train)))
    
	log.info(f"Confusion matrix: \n{confusion_matrix(y_test, y_pred)}")
    # print(confusion_matrix(y_test, y_pred))
    
	if test_score > max_score:
		log.info(f"New max score: {test_score}, model: {model_name}")
		max_score = test_score
		max_model_name = model_name
		max_model = model
    


# ### Create a logistic regression model, then call train_test to implement the steps explained above
log.info("Run Logistic Regression")
lr = LogisticRegression()    
train_test(lr, "Logistic Regression", X_train, X_test, y_train, y_test)


# ### Create a Random Forest Classifier model, then call train_test to implement the steps explained above
log.info("Random Forest")    
rf = RandomForestClassifier(n_estimators=300, criterion="gini", max_depth=10, n_jobs=5) 
train_test(rf, "Random Forest", X_train, X_test, y_train, y_test)


# ### Create a Support Vector - RBF model, then call train_test to implement the steps explained above
log.info("Support Vector - RBF")    
svc = svm.SVC(kernel="rbf",max_iter=-1,C=10**9, gamma="auto")
train_test(svc, "Support Vector - RBF", X_train, X_test, y_train, y_test)


# ### Create a Naive Bayes model, then call train_test to implement the steps explained above
log.info("Naive Bayes")
scaler = MinMaxScaler()
fit = scaler.fit(X_train)
X_train_m = fit.transform(X_train)
X_test_m = fit.transform(X_test)

mnb = MultinomialNB()
train_test(mnb, "Naive Bayes", X_train_m, X_test_m, y_train, y_test)


# ### Create a K Nearest Neighbour model, then call train_test to implement the steps explained above
log.info("K Nearest Neighbour clustering")
knn = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1)
train_test(knn, "K Nearest Neighbour", X_train, X_test, y_train, y_test)


# ### Create a Support Vector - Linear model, then call train_test to implement the steps explained above
log.info("Support Vector Machines - Linear")    
svm=LinearSVC(C=0.0001)
train_test(svm, "Support Vector - Linear", X_train, X_test, y_train, y_test)


# ### Create a Decision Trees model, then call train_test to implement the steps explained above
log.info("Decision Trees Classifier")    
clf = DecisionTreeClassifier()
train_test(clf, "Decision Trees",X_train, X_test, y_train, y_test)


# ### Create a Bagging Classifier model, then call train_test to implement the steps explained above
log.info("Bagging Classifier")    
bg=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=1000)
train_test(bg, "Bagging", X_train, X_test, y_train, y_test)


# ### Create a AdaBoost Classifier model, then call train_test to implement the steps explained above
log.info("AdaBoost Classifier")    
adb = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4),n_estimators=1000,learning_rate=0.6)
train_test(bg, "AdaBoost", X_train, X_test, y_train, y_test)


# ### Create a Tensorflow Neural Network model, then call train_test to implement the steps explained above
log.info("Neural Network")
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout

log.info("Creating neural net")
x_partial_train, x_validation, y_partial_train, y_validation = train_test_split(X_train, y_train, test_size=0.3)
model=models.Sequential()
model.add(layers.Dense(4096,activation='relu',input_shape=(14,)))
model.add(Dropout(0.2))
model.add(layers.Dense(2048,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(1024,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(1,activation='sigmoid'))

sgd = optimizers.SGD(lr=0.01)

model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])  # rmsprop

log.info("Training neural net model")
model.fit(x_partial_train,y_partial_train,epochs=150,validation_data=(x_validation,y_validation),verbose=0)
log.info("score on test: " + str(model.evaluate(X_test,y_test)[1]))

y_pred = model.predict(X_test)
y_pred = [0 if y < 0.5 else 1 for y in y_pred]

y_test.reset_index(drop=True,inplace=True)
log.info(f"Confusion matrix: \n{confusion_matrix(y_test, y_pred)}")

test_score = sum(y_pred == y_test)/len(y_test)
log.info(f"Neural Net score: {test_score}")

if test_score > max_score:
	max_score = test_score
	max_model_name = "Neural Network"
	max_model = model
	log.info(f"New max score: {test_score}, model: {max_model_name}")


# ### Save the mode so it can be retrieved by the prediction service
filename = f"{pn_config.project_path}/{pn_config.pickled_model_file}"

log.info(f"Save best-performing model, {max_model_name}, to disk: {filename}")
with open(filename,'wb') as f:
    pickle.dump(max_model, f)

