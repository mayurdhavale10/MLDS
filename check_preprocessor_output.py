import pickle
import pandas as pd

# Load the preprocessor
preprocessor_path = "artifacts/preprocessor.pkl"  # Ensure the path is correct

try:
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    # ✅ Define a sample test input using correct column names
    sample_data = pd.DataFrame([{
        "gender": "male",
        "race/ethnicity": "group B",  # Ensure correct name
        "parental level of education": "bachelor's degree",  # Ensure correct name
        "lunch": "standard",
        "test preparation course": "none",
        "reading score": 72,  # Ensure correct name (not reading_score)
        "writing score": 74   # Ensure correct name (not writing_score)
    }])

    # Transform the sample data
    transformed_data = preprocessor.transform(sample_data)

    # ✅ Print the transformed data and its shape
    print("\n✅ Transformed Data:", transformed_data)
    print("🔍 Number of Features After Transformation:", transformed_data.shape[1])

except Exception as e:
    print(f"❌ Error loading preprocessor or transforming data: {e}")
