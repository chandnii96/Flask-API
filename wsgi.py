from app import app
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging

if __name__ == "__main__":
    prediction = None
    df = pd.read_csv("F:/Projects/crop-recommendation-system-based-on-machine-learning-using-python-master/Data/crop_recommendation.csv")
    features = df[['N','P','K','temperature','humidity']]
    target = df['label']
    RF = RandomForestClassifier(n_estimators=29, criterion = 'entropy',random_state=0)
    RF.fit(features, target)

    app.run(debug=True,host='127.0.0.1', port=5001)