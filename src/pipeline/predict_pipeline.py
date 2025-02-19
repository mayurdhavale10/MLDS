import os
import sys
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class PredictPipelineConfig:
    """Define paths for the trained model and preprocessor."""
    model_file_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "model.pkl"))
    preprocessor_file_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "preprocessor.pkl"))


class CustomData:
    """Handles input data and converts it into a DataFrame with correct column names."""
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """Convert input data into a correctly formatted DataFrame."""
        df = pd.DataFrame({
            'gender': [self.gender],
            'race/ethnicity': [self.race_ethnicity],
            'parental level of education': [self.parental_level_of_education],
            'lunch': [self.lunch],
            'test preparation course': [self.test_preparation_course],
            'reading score': [self.reading_score],
            'writing score': [self.writing_score]
        })

        print("\n🔍 Generated DataFrame:")
        print(df)
        return df


class PredictPipeline:
    """Loads the model and preprocessor to make predictions on input data."""
    def __init__(self):
        self.config = PredictPipelineConfig()

    def predict(self, features: pd.DataFrame):
        try:
            print("\n📂 Loading Model and Preprocessor...")
            model = load_object(self.config.model_file_path)
            preprocessor = load_object(self.config.preprocessor_file_path)

            logging.info("✅ Model and Preprocessor Loaded Successfully!")

            # Ensure the model receives the expected columns
            expected_columns = list(preprocessor.get_feature_names_out())
            received_columns = list(features.columns)

            print("\n🔎 Expected Input Columns for Preprocessing:", expected_columns)
            print("🔎 Received Columns in Input:", received_columns)

            # Verify column names match
            missing_cols = set(expected_columns) - set(received_columns)
            if missing_cols:
                raise ValueError(f"❌ ERROR: Missing columns in input data: {missing_cols}")

            print("\n✅ Input Data Matches Expected Columns! Proceeding with transformation...")

            # Pass already transformed data (since app.py does it)
            transformed_features = features  # ✅ Already preprocessed

            print("\n✅ Transformed Data Shape:", transformed_features.shape)
            print("✅ Transformed Data:\n", transformed_features)

            # Make Predictions
            predictions = model.predict(transformed_features)

            print("\n🎯 Prediction Completed! Prediction:", predictions)
            return predictions

        except Exception as e:
            logging.error(f"❌ Error during prediction: {e}")
            raise CustomException(e, sys)
