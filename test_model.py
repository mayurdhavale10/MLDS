import pickle
import numpy as np
import pandas as pd

# Load preprocessor and model
preprocessor_path = "artifacts/preprocessor.pkl"
model_path = "artifacts/model.pkl"

try:
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("✅ Preprocessor & Model Loaded Successfully!")

    # ✅ FIX: Column names exactly as seen in training data
    test_input = pd.DataFrame([{
        "gender": "male",
        "race/ethnicity": "group C",   # ✅ Fix: Match exact column name
        "parental level of education": "bachelor's degree",  # ✅ Fix
        "lunch": "standard",
        "test preparation course": "none",  # ✅ Fix
        "reading score": 80,  # ✅ Fix: Match exactly
        "writing score": 90   # ✅ Fix: Match exactly
    }])

    print("\n🔍 Test Input Before Transformation:")
    print(test_input)

    # Apply preprocessing
    transformed_input = preprocessor.transform(test_input)
    print("\n✅ Transformed Input (Shape:", transformed_input.shape, "):")
    print(transformed_input)

    # Check if transformed features match expected input size
    if transformed_input.shape[1] != 19:
        print(f"\n❌ Error: Preprocessed input has {transformed_input.shape[1]} features, but model expects 19.")
    else:
        # Make prediction
        prediction = model.predict(transformed_input)
        print("\n✅ Model Prediction:", prediction)

except Exception as e:
    print("\n❌ Error:", e)
