from flask import Flask, request, render_template
import os
import sys
import pickle
import pandas as pd
import numpy as np

print("\n✅ Initializing Flask App...")

# Ensure `src` is correctly included in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
print("🔄 Updated Python Path for `src`")

# Import necessary modules
try:
    print("📂 Importing PredictPipeline...")
    from src.pipeline.predict_pipeline import CustomData, PredictPipeline
    print("✅ PredictPipeline Imported Successfully!")
except Exception as e:
    print(f"❌ Error Importing PredictPipeline: {e}")

# Initialize Flask app
application = Flask(__name__)
app = application

# Load the preprocessor at the start
preprocessor_path = "artifacts/preprocessor.pkl"
with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

print("\n✅ Preprocessor Loaded Successfully!")

# Display expected feature names from preprocessor
expected_features = preprocessor.get_feature_names_out()
print("\n🔎 Expected Features in Preprocessor:", expected_features)

# Home route
@app.route('/')
def index():
    print("🚀 Home Route Accessed")
    return render_template('index.html')

# Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    print("\n🚀 Predict Route Accessed")

    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            print("📝 Extracting Form Data...")
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            print("📊 Converting Form Data to DataFrame...")
            pred_df = data.get_data_as_data_frame()
            print("\n🔍 Input Data Before Transformation:")
            print(pred_df)

            print("\n🔄 Applying Preprocessor Transformation...")
            transformed_pred_df = preprocessor.transform(pred_df)

            # Convert transformed array back to DataFrame with correct column names
            transformed_pred_df = pd.DataFrame(transformed_pred_df, columns=expected_features)
            print("\n✅ Transformed Data After Renaming:")
            print(transformed_pred_df)

            print("⚡ Before Prediction")
            predict_pipeline = PredictPipeline()
            print("⚡ Mid Prediction")

            results = predict_pipeline.predict(transformed_pred_df)  # Pass correctly formatted DataFrame
            print("⚡ After Prediction 🎯")

            return render_template('home.html', results=results[0])

        except Exception as e:
            print(f"❌ Error During Prediction: {e}")
            return render_template('home.html', error="Something went wrong!")

if __name__ == "__main__":
    print("\n🚀 Flask App is Starting...")

    try:
        PORT = 5000  # Change this if needed
        print(f"🌍 Attempting to start Flask server on port {PORT}...")
        app.run(host="0.0.0.0", port=PORT, debug=True)
        print("✅ Flask server started successfully!")

    except Exception as e:
        print(f"❌ Error Starting Flask: {e}")
