import sys
import os
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["CRS_DEP_TIME"]
            categorical_columns = ["QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", 
                                   "OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]

            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, numerical_columns),
                 ("cat_pipelines", cat_pipeline, categorical_columns)]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Shape of original train dataframe: {train_df.shape}")
            logging.info(f"Shape of original test dataframe: {test_df.shape}")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "DEP_DELAY_NEW"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Inspecting input features before transformation (train)")
            logging.info(input_feature_train_df.head())

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info("Shape of train features after one hot encoding: {}".format(input_feature_train_arr.shape))

            if sparse.issparse(input_feature_train_arr):
                logging.info("Transformed train features are in sparse format, converting to dense.")
                input_feature_train_arr = input_feature_train_arr.toarray()

            logging.info(f"Shape after transformation (train features): {input_feature_train_arr.shape}")

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Shape of test features after one hot encoding: {}".format(input_feature_test_arr.shape))

            if sparse.issparse(input_feature_test_arr):
                logging.info("Transformed test features are in sparse format, converting to dense.")
                input_feature_test_arr = input_feature_test_arr.toarray()

            logging.info(f"Shape after transformation (test features): {input_feature_test_arr.shape}")

            if input_feature_train_arr.shape[0] != target_feature_train_df.shape[0]:
                raise ValueError("Mismatch in train feature and target array sizes after transformation")

            if input_feature_test_arr.shape[0] != target_feature_test_df.shape[0]:
                raise ValueError("Mismatch in test feature and target array sizes after transformation")

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion  # Ensure this import is correct based on your project structure
    obj = DataIngestion()
    obj.initiate_data_ingestion()
