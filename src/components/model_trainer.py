import os
import sys
import pickle
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, prepare_dense_data

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    neural_network_model_file_path = os.path.join("artifacts", "neural_network_model.h5")

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

            X_train_dense, X_test_dense = prepare_dense_data(X_train, X_test)

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                #"Random Forest": self.train_random_forest(X_train, y_train),
                "XGBRegressor": XGBRegressor(),
                "LightGBM Regressor": LGBMRegressor(),
                "Neural Network": self.train_neural_network(X_train_dense, y_train, X_test_dense, y_test)
            }

            model_mae_scores = {}

            for model_name, model in models.items():
                if model_name != "Neural Network":
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict(X_test_dense).flatten()

                mae = mean_absolute_error(y_test, y_pred)
                model_mae_scores[model_name] = mae
                logging.info(f"{model_name} MAE: {mae}")

            best_model_name = min(model_mae_scores, key=model_mae_scores.get)
            best_model = models[best_model_name]
            logging.info(f"Best performing model: {best_model_name} with MAE: {model_mae_scores[best_model_name]}")

            # Check if the best model is a RandomForestRegressor and log its hyperparameters
            if best_model_name == "Random Forest":
                best_rf_params = best_model.get_params()
                logging.info(f"Hyperparameters of the best RandomForestRegressor: {best_rf_params}")

            # Saving the best model
            if best_model_name == "Neural Network":
                best_model.save(self.model_trainer_config.neural_network_model_file_path)
            else:
                save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)


            return models

        except Exception as e:
            raise CustomException(str(e), sys.exc_info())

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

    def train_random_forest(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 500],
        }
        rf_grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=2, scoring='neg_mean_absolute_error')
        rf_grid_search.fit(X_train, y_train)
        return rf_grid_search.best_estimator_