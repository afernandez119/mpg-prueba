# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:14:13 2020

@author: 0A00066
"""
import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_mpg

app = Flask('app')

@app.route('/', methods=['GET'])
def index():
    return 'Pinging Model Application!!'


@app.route('/predict',methods=['POST'])
def predict():
    vehicle = request.get_json()
    print(vehicle)
    
    with open('./model_files/model.bin','rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
        
    predictions = predict_mpg(model, vehicle)
    
    result = {
        'mpg_prediction' : list(predictions)}
    
    return jsonify(result)

#Descomentar solo para prueba
#if __name__=='__main__':
#    app.run(debug=False, host='localhost', port=4996)
    
    
    