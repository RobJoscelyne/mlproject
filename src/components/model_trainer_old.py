from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_neural_network(self, X_train, y_train, X_test, y_test):
        nn_model = Sequential([
            Dense(128, input_dim=X_train.shape[1], activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])

        adam_optimizer = Adam(learning_rate=1e-3)
        nn_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=128, callbacks=[early_stopping])

        return nn_model

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
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "LightGBM Regressor": LGBMRegressor()
            }

            # Evaluate models using MAE
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, scoring='neg_mean_absolute_error')

            # Train and evaluate the neural network
            logging.info("Training Neural Network")
            nn_model = self.train_neural_network(X_train, y_train, X_test, y_test)

            # Predict and evaluate using the neural network
            nn_predictions = nn_model.predict(X_test)
            nn_mae = mean_absolute_error(y_test, nn_predictions)
            model_report['Neural Network'] = -nn_mae  # Assuming MAE should be negative for consistency with other models

            best_model_score = max(model_report.values())
            best_model_name = [model_name for model_name, score in model_report.items() if score == best_model_score][0]
            best_model = models.get(best_model_name, nn_model)

            if -best_model_score > 20:  # Assuming the scoring returns negative MAE
                raise CustomException("No best model found")

            logging.info(f"Best model found on both training and testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test) if best_model_name != 'Neural Network' else nn_predictions
            mae = mean_absolute_error(y_test, predicted)
            return best_model_name, mae

        except Exception as e:
            raise CustomException(e, sys)


