#!/usr/bin/env python
# coding: utf-8

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

import pn_config
import logging

# Load the logging configuration
logging.basicConfig(filename=f"{pn_config.log_file}", filemode='w', format=f'{pn_config.log_format}', level = logging.DEBUG)
log = logging.getLogger("Predict_Now_predict")

# Flask configuration
app = Flask(__name__)

# Load the saved model
filename = "predict_now.sav"
infile = open(filename, 'rb')
model = pickle.load(infile, encoding='bytes')
log.info("Loaded ML model")

# The columns in data file
params = ['dp', 'dy', 'ep', 'de', 'svar', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl']

# This function displays the user interface
@app.route("/")
def index():
	log.info("Load index page")
	return render_template("index.html")
	
# This function clears the contents of the user interface
@app.route("/clear", methods=['POST'])
def clear():
	log.info("Clear index page")
	return render_template('index.html', raw_text="", prediction_result="")

# This function expects a json object containing the values of
# columns for which a prediction is to be made and returns the
# prediction are a json object.
@app.route("/process", methods=['POST'])
def process():
	log.info("Get Returns target from predictor variables")
	
	results = ""
	try:
		json_text = request.form["rawtext"]	
		json1_data = json.loads(json_text)
			
		d = dict()
		
		for param in params:
			key = param
			val = []
			val.append(json1_data[key])
			d[key] = val
			
		df = pd.DataFrame(d)
		
		y_pred = model.predict(df)
		
		results = [str(y_pred[0])]
		
	except Exception:
		log.error("Illegal data submitted") 
		results = ["Illegal data submitted"]
		
		
	return jsonify(Returns = results)