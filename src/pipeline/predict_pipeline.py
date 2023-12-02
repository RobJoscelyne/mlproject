import os
import pandas as pd
import pickle
from src.utils import load_object
from src.exception import CustomException

class PredictPipeline:
    def __init__(self, model_path=None, preprocessor_path=None):
        self.model_path = model_path or 'C:\\Users\\robjo\\mlproject\\artifacts\\model1.pkl'
        self.preprocessor_path = preprocessor_path or 'C:\\Users\\robjo\\mlproject\\artifacts\\proprocessor.pkl'

    def predict(self, features):
        try:
            print("Loading model from:", self.model_path)
            with open(self.model_path, 'rb') as file:
                model = pickle.load(file)

            # Workaround for the gpu_id attribute error in XGBoost model
            if hasattr(model, 'get_xgb_params'):
                original_get_params = model.get_params

                def patched_get_params(deep=True): 
                    try:
                        return original_get_params(deep=deep)
                    except AttributeError as e:
                        if str(e) == "'XGBModel' object has no attribute 'gpu_id'":
                            return {}
                        raise e

                model.get_params = patched_get_params

            print("Model loaded successfully.")

            print("Loading preprocessor from:", self.preprocessor_path)
            with open(self.preprocessor_path, 'rb') as file:
                preprocessor = pickle.load(file)
            print("Preprocessor loaded successfully.")

            print("Preprocessing features...")
            data_scaled = preprocessor.transform(features)
            print("Features preprocessed.")

            print("Making predictions...")
            preds = model.predict(data_scaled)
            print("Predictions made.")

            return preds

        except FileNotFoundError as e:
            raise CustomException(f"A file was not found: {e.filename}", e)
        except Exception as e:
            raise CustomException("An error occurred in PredictPipeline", e)

class CustomData:
    def __init__(self, quarter: str, month: str, day_of_month: str, 
                 day_of_week: str, op_unique_carrier: str, 
                 origin: str, dest: str, crs_dep_time: int):
        
        self.data = {
            "QUARTER": quarter,
            "MONTH": month,
            "DAY_OF_MONTH": day_of_month,
            "DAY_OF_WEEK": day_of_week,
            "OP_UNIQUE_CARRIER": op_unique_carrier,
            "ORIGIN": origin,
            "DEST": dest,
            "CRS_DEP_TIME": crs_dep_time,
        }

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame([self.data])
        except Exception as e:
            raise CustomException(f"An error occurred in CustomData: {e}")

if __name__ == "__main__":
    custom_data = CustomData("3", "8", "6", "6", "B6", "JFK", "LAX", 1920)
    pipeline = PredictPipeline()
    prediction = pipeline.predict(custom_data.get_data_as_dataframe())
    print("Prediction:", prediction)
