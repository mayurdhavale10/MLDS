# import sys
# import os
# from dataclasses import dataclass
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# # Adjusted import paths to directly reference the modules
# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object

# @dataclass
# class DataTransformationConfig:
#     # Set the path for the preprocessor object file relative to the script's directory
#     preprocessor_obj_file_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts', "preprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformer_object(self):
#         """
#         This function is responsible for data transformation.
#         """
#         try:
#             # Define numerical and categorical columns
#             numerical_columns = ['reading score', 'writing score']
#             categorical_columns = [
#                 "gender",
#                 "race/ethnicity",
#                 "parental level of education",
#                 "lunch",
#                 "test preparation course",
#             ]

#             # Create numerical transformation pipeline
#             num_pipeline = Pipeline(
#                 steps=[
#                     ("imputer", SimpleImputer(strategy="median")),
#                     ("scaler", StandardScaler())
#                 ]
#             )

#             # Create categorical transformation pipeline
#             cat_pipeline = Pipeline(
#                 steps=[
#                     ("imputer", SimpleImputer(strategy="most_frequent")),
#                     ("one_hot_encoder", OneHotEncoder()),
#                     ("scaler", StandardScaler(with_mean=False))
#                 ]
#             )

#             # Log information about the columns
#             logging.info(f"Categorical columns: {categorical_columns}")
#             logging.info(f"Numerical columns: {numerical_columns}")

#             # Combine both pipelines into a ColumnTransformer
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("num_pipeline", num_pipeline, numerical_columns),
#                     ("cat_pipeline", cat_pipeline, categorical_columns)
#                 ]
#             )

#             return preprocessor
        
#         except Exception as e:
#             raise CustomException(e, sys)
        
#     def initiate_data_transformation(self, train_path, test_path):
#         try:
#             # Read the train and test data
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info("Read train and test data completed")
#             logging.info("Obtaining preprocessing object")

#             # Get the preprocessing object
#             preprocessing_obj = self.get_data_transformer_object()

#             # Define target and feature columns
#             target_column_name = "math score"  # Updated to match the column name in your dataset
#             numerical_columns = ["writing score", "reading score"]  # Ensure correct column names

#             input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
#             target_feature_train_df = train_df[target_column_name]

#             input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
#             target_feature_test_df = test_df[target_column_name]

#             logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
#             # Fit and transform the training data, transform the test data
#             input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

#             # Combine features and target variables
#             train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             logging.info("Saving preprocessing object.")
#             # Save the preprocessing object
#             save_object(
#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj
#             )

#             logging.info(f"Preprocessing object saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

#             return (
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path,
#             )
            
#         except Exception as e:
#             raise CustomException(e, sys)

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import necessary custom modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # Set the path for the preprocessor object file relative to the script's directory
    preprocessor_obj_file_path: str = os.path.join(os.getcwd(), 'artifacts', "preprocessor.pkl")

    # preprocessor_obj_file_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ['reading score', 'writing score']
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            # Create numerical transformation pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Create categorical transformation pipeline with `handle_unknown="ignore"`
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),  # ✅ FIXED
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Log information about the columns
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Define target and feature columns
            target_column_name = "math score"  # Ensure correct column name
            numerical_columns = ["writing score", "reading score"]

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Fit and transform the training data, transform the test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target variables
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object.")
            
            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessing object saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)
