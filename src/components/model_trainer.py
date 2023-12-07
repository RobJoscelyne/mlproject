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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

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
            }

            lgbm_param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100],
                'max_depth': [10, 20, 30]
            }

            logging.info("Starting Grid Search for LightGBM")
            lgbm_grid_search = GridSearchCV(LGBMRegressor(), lgbm_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            lgbm_grid_search.fit(X_train, y_train)

            best_lgbm = lgbm_grid_search.best_estimator_
            best_lgbm_params = lgbm_grid_search.best_params_
            logging.info(f"Best parameters for LightGBM: {best_lgbm_params}")
            models['LightGBM Regressor'] = best_lgbm

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

            model_file_path = self.model_trainer_config.trained_model_file_path
            save_object(file_path=model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, predicted)
            return best_model_name, mae

        except Exception as e:
            raise CustomException(e, sys)

# Further implementation of your project...
