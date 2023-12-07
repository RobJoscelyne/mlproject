import os
import sys
import numpy as np 
import pandas as pd
import pickle
import dill
from sklearn.metrics import r2_score, mean_absolute_error, get_scorer
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

from tensorflow.keras.models import Model

def save_object(file_path, obj):
    try:
        if isinstance(obj, Model):
            # For Keras models
            obj.save(file_path)
        else:
            # For other objects
            with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, scoring='r2'):
    try:
        report = {}
        scoring_function = get_scorer(scoring)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            if scoring == 'r2':
                score = r2_score(y_test, y_test_pred)
            elif scoring == 'neg_mean_absolute_error':
                score = -mean_absolute_error(y_test, y_test_pred)
            else:
                score = scoring_function(y_test, y_test_pred)

            report[model_name] = score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)