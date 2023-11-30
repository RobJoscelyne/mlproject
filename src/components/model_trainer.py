# In your model_trainer.py file
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

# In your model_trainer.py file
from src.utils import save_object, evaluate_models, prepare_dense_data

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, prepare_dense_data

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    neural_network_model_file_path=os.path.join("artifacts", "neural_network_model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Prepare data for the neural network (if needed)
            X_train_dense, X_test_dense = prepare_dense_data(X_train, X_test)

            # Define traditional models
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "LightGBM Regressor": LGBMRegressor(),
                "Random Forest": RandomForestRegressor(),
                # Add neural network model
                "Neural Network": self.train_neural_network(X_train_dense, y_train, X_test_dense, y_test)
            }

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )

            # Define a threshold for MAE
            acceptable_mae_threshold = 20  # Example threshold, adjust according to your needs

            # Finding the best model based on lowest MAE
            best_model_name = min(model_report, key=model_report.get)
            best_model_mae = model_report[best_model_name]

            if best_model_mae > acceptable_mae_threshold:
                raise CustomException("No suitable model found with low MAE", sys.exc_info())

            logging.info(f"Best model found: {best_model_name} with MAE: {best_model_mae}")

            # Save the best model, handle neural network separately
            if best_model_name == "Neural Network":
                models[best_model_name].save(self.model_trainer_config.neural_network_model_file_path)
            else:
                best_model = models[best_model_name]
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

            return best_model_mae

        except Exception as e:
            raise CustomException(str(e), sys.exc_info())

    def train_neural_network(self, X_train, y_train, X_test, y_test):
        # Define the Neural Network with Dropout
        nn_model = Sequential([
            Dense(128, input_dim=X_train.shape[1], activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])

        # Compile and fit the model
        adam_optimizer = Adam(learning_rate=1e-3)
        nn_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=128, callbacks=[early_stopping])

        return nn_model
