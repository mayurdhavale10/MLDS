import pickle
import pandas as pd
import pickle
import pandas as pd

# ✅ Load the preprocessor from the saved pickle file
preprocessor_path = "artifacts/preprocessor.pkl"  # Ensure this path is correct

try:
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    print("✅ Preprocessor Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading preprocessor: {e}")
    exit()

# ✅ Define test input data
test_df = pd.DataFrame({
    'gender': ['male'],
    'race/ethnicity': ['group B'],
    'parental level of education': ["bachelor's degree"],
    'lunch': ['standard'],
    'test preparation course': ['completed'],
    'reading score': [80],
    'writing score': [90]
})

print("\n🔍 Input Data Before Transformation:\n", test_df)

# ✅ Transform the data
try:
    transformed = preprocessor.transform(test_df)
    print("\n✅ Transformed Data Shape:", transformed.shape)
    print("✅ Transformed Data:\n", transformed)
except Exception as e:
    print(f"\n❌ Error during transformation: {e}")
