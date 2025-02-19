import pickle
import pandas as pd

# ✅ Load Preprocessor
preprocessor_path = "artifacts/preprocessor.pkl"
with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

print("\n✅ Expected Features in Preprocessor:")
print(preprocessor.feature_names_in_)  # Make sure this matches test_df columns

# ✅ Define Test Sample (Ensure Columns Match Preprocessor Expectations)
test_df = pd.DataFrame({
    'gender': ['male'],
    'race/ethnicity': ['group B'],
    'parental level of education': ["bachelor's degree"],
    'lunch': ['standard'],
    'test preparation course': ['completed'],
    'reading score': [80],
    'writing score': [90]
})

print("\n🔍 Input Data Before Transformation:")
print(test_df)

# ✅ Apply Transformation
transformed_data = preprocessor.transform(test_df)

print("\n✅ Transformed Data Shape:", transformed_data.shape)
print("✅ Transformed Data:", transformed_data)
