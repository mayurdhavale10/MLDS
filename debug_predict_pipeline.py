import pickle
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# ✅ Define a sample test case (Ensure this matches the model's expected input)
test_input = CustomData(
    gender="male",
    race_ethnicity="group C",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="none",
    reading_score=80,
    writing_score=90
)

# Convert to DataFrame
test_df = test_input.get_data_as_data_frame()
print("\n🔍 Test Data Before Transformation:\n", test_df)

# ✅ Load the preprocessor separately and test transformation
preprocessor_path = "artifacts/preprocessor.pkl"
with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

print("\n🔎 Expected Columns in Preprocessor:")
expected_columns = preprocessor.feature_names_in_
print(expected_columns)

# Validate and Transform
if set(expected_columns) == set(test_df.columns):
    transformed_input = preprocessor.transform(test_df)
    print("\n✅ Transformed Test Data (Shape: {}):\n".format(transformed_input.shape), transformed_input)
else:
    print("\n❌ Column mismatch! Expected:", expected_columns, "but got:", test_df.columns)

# ✅ Run Prediction
predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(test_df)

print("\n🚀 **Final Prediction:**", prediction)
