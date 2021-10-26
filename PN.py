#!/usr/bin/env python
# coding: utf-8


# import xlsxwriter
import pylightxl as xl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle



# readxl returns a pylightxl database that holds all worksheets and its data
db = xl.readxl(fn='SPX_train_0.xlsx')


file_rows = []

for row in db.ws(ws='Sheet1').rows:
       file_rows.append(row)


df = pd.DataFrame(file_rows[1:])
df.columns = file_rows[0]

corr_df = df.corr()

returns = corr_df["Returns"]

five_percent_corr = returns[abs(returns) > 0.05 ]


X = df[[c for c in df.columns if c in five_percent_corr]]
y = df["Returns"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)


print(confusion_matrix(y_test, y_pred))


filename = "predict_now.sav"
pickle.dump(lr, open(filename, 'wb'))
