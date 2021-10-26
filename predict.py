#!/usr/bin/env python
# coding: utf-8


# import xlsxwriter
import pylightxl as xl
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

lr = pickle.load(infile, encoding='bytes')

params = ['dp', 'dy', 'ep', 'de', 'svar', 'bm', 'ntis', 'tbl', 'lty', 'ltr',
       'tms', 'dfy', 'dfr', 'infl']


@app.route("/")
def index():
	return render_template("index.html")
	
@app.route("/process", methods=['GET', 'POST'])
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
	
	y_pred = lr.predict(df)
	
	
	print(y_pred)
	
	# results = [("prediction_result",y_pred)]
	
	# response = make_response(render_template('index.html',"prediction_result"=y_pred))
	# response.headers["prediction_result"] = y_pred
	# return response
	# response = app.response_class(
		
	js = [ { "prediction_result" : int(y_pred[0]) } ]
	
	response =  Response(json.dumps(js),  mimetype='application/json')
	return render_template('index.html', prediction_result=int(y_pred[0]))