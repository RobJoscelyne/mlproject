import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    artifacts_dir = "artifacts"
    trained_model_file_path = os.path.join(artifacts_dir, "model.pkl")
    nn_model_file_path = os.path.join(artifacts_dir, "neural_network_model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        if not os.path.exists(self.model_trainer_config.artifacts_dir):
            os.makedirs(self.model_trainer_config.artifacts_dir)
    
    def create_neural_network(self, input_dim):
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            BatchNormalization(),  # Batch normalization after the Dense layer
            Dropout(0.2),
            Dense(128, activation='relu'),
            BatchNormalization(),  # Batch normalization after the Dense layer
            Dropout(0.1),
            Dense(128, activation='relu'),
            BatchNormalization(),  # Batch normalization after the Dense layer
            Dropout(0.1),
            Dense(32, activation='relu'),
            BatchNormalization(),  # Batch normalization after the Dense layer
            Dropout(0.1),
            Dense(16, activation='relu'),
            BatchNormalization(),  # Batch normalization after the Dense layer
            Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def train_neural_network(self, model, X_train, y_train, X_test, y_test):
        checkpoint_cb = ModelCheckpoint(self.model_trainer_config.nn_model_file_path, save_best_only=True)
        early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10)

        batch_size = 256
        model.fit(X_train, y_train, batch_size=batch_size, epochs=200, validation_data=(X_test, y_test), callbacks=[checkpoint_cb, early_stopping_cb])

        loaded_model = tf.keras.models.load_model(self.model_trainer_config.nn_model_file_path)
        predictions = loaded_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        logging.info(f"Neural Network model MAE: {mae}")

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "LightGBM Regressor": LGBMRegressor(),
                "Neural Network": self.create_neural_network(X_train.shape[1])  # Adding neural network
            }

            self.train_neural_network(models["Neural Network"], X_train, y_train, X_test, y_test)
            del models["Neural Network"]  # Remove Neural Network from the models dictionary

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, scoring='neg_mean_absolute_error')

            sorted_models = sorted(model_report.items(), key=lambda x: x[1], reverse=True)
            top_5_models = sorted_models[:5]
            logging.info("Top 5 models:")
            for model_name, score in top_5_models:
                logging.info(f"{model_name}: {-score}")

            best_model_name, best_model_score = top_5_models[0]
            best_model = models[best_model_name]

            if -best_model_score > 20:
                raise CustomException("No best model found")

            logging.info(f"Best model found on both training and testing dataset: {best_model_name}")

            if best_model_name != "Neural Network": 
                model_file_path = self.model_trainer_config.trained_model_file_path
                save_object(file_path=model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, predicted)
            return best_model_name, mae

        except Exception as e:
            raise CustomException(e, sys)

# Further implementation of your project...
