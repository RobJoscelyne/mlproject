import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            # Correct path for the preprocessor
            preprocessor_path = os.path.join('artifacts', 'proprocessor.pkl')  # Updated path

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, CRS_DEP_TIME: int, MONTH: int, DAY_OF_WEEK: int, 
                 OP_UNIQUE_CARRIER: str, ORIGIN: str, DEST: str):

        self.CRS_DEP_TIME = CRS_DEP_TIME
        self.MONTH = MONTH
        self.DAY_OF_WEEK = DAY_OF_WEEK
        self.OP_UNIQUE_CARRIER = OP_UNIQUE_CARRIER
        self.ORIGIN = ORIGIN
        self.DEST = DEST

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CRS_DEP_TIME": [self.CRS_DEP_TIME],
                "MONTH": [self.MONTH],
                "DAY_OF_WEEK": [self.DAY_OF_WEEK],
                "OP_UNIQUE_CARRIER": [self.OP_UNIQUE_CARRIER],
                "ORIGIN": [self.ORIGIN],
                "DEST": [self.DEST],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
