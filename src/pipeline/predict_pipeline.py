import sys
import os
import pandas as pd
import tensorflow as tf
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        base_path = "C:\\Users\\robjo\\mlproject\\artifacts"
        nn_model_path = os.path.join(base_path, "neural_network_model.h5")
        self.model = tf.keras.models.load_model(nn_model_path)  # Load model once during initialization
        self.preprocessor_path = os.path.join(base_path, 'proprocessor.pkl')  # Path to the preprocessor

    def predict(self, features):
        try:
            preprocessor = load_object(file_path=self.preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = self.model.predict(data_scaled)  # Use the pre-loaded model
            return preds.squeeze()
        
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
