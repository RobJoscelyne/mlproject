import os
import sys
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def prepare_dense_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            if model_name == "Neural Network":
                y_test_pred = model.predict(X_test).flatten()
            else:
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_test_pred)
            report[model_name] = mae
        return report
    except Exception as e:
        raise CustomException(e, sys.exc_info())
