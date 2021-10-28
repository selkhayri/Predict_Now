#!/usr/bin/env python
# coding: utf-8


# import xlsxwriter
# import pylightxl as xl
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
import json
from flask import Flask, request,make_response,render_template,url_for,\
	request,jsonify,Response


app = Flask(__name__)


# readxl returns a pylightxl database that holds all worksheets and its data


filename = "predict_now.sav"
infile = open(filename, 'rb')

model = pickle.load(infile, encoding='bytes')

params = ['dp', 'dy', 'ep', 'de', 'svar', 'bm', 'ntis', 'tbl', 'lty', 'ltr',
       'tms', 'dfy', 'dfr', 'infl']


@app.route("/")
def index():
	return render_template("index.html")
	
@app.route("/clear", methods=['POST'])
def clear():
	print("Clear")
	return render_template('index.html', raw_text="", prediction_result="")

@app.route("/process", methods=['POST'])
def process():
	
	json_text = request.form["rawtext"]
		
	json1_data = json.loads(json_text)
	
	# json_dict = json.decoder.JSONObject(json_text)
	d = dict()
	
	
	for param in params:
		key = param
		val = []
		val.append(json1_data[key])
		d[key] = val
		
	df = pd.DataFrame(d)
	
	y_pred = model.predict(df)
	
	return render_template('index.html', raw_text=json_text, prediction_result=int(y_pred[0]))