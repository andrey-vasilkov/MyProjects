import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import cv2
import time


class Model:
    def __init__(self):
        model_path = os.path.join('data', 'model.pkl')
        self.model = joblib.load(model_path)
        self.labels_path = os.path.join("data", "emnist-balanced-mapping.txt")
        self.dict_labels = self.load_dict_labels()

    def load_dict_labels(self):

        with open(self.labels_path) as file:
            text = file.read()
        dict_labels = {}

        for line in text.splitlines():
            label = int(line.split()[0])
            code = int(line.split()[1])
            dict_labels[label] = chr(code)

        return dict_labels
        # your code here


    def predict( self, x:np.ndarray):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        # your code here
        x1 = x.reshape(1, -1)
        prediction = self.model["model"].predict(x1)[0]

        return self.dict_labels[prediction]


