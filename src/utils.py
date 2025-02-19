import os
import sys
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj (object): The object to save.
    """
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple regression models and return their R² scores.
    
    Parameters:
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target variable.
    - X_test (np.ndarray): Testing features.
    - y_test (np.ndarray): Testing target variable.
    - models (dict): Dictionary of model names and instances.
    - params (dict): Dictionary of model names and their corresponding hyperparameter grids.

    Returns:
    - dict: A dictionary containing model names and their R² test scores.
    """
    try:
        report = {}

        for model_name, model in models.items():
            # Set hyperparameters using GridSearchCV if applicable
            if params[model_name]:  # Check if there are hyperparameters to tune
                gs = GridSearchCV(model, params[model_name], cv=3)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_  # Use the best model found
            else:
                model.fit(X_train, y_train)  # Train the model without tuning

            # Predictions on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R² scores for training and testing data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            logging.info(f"{model_name} - Train R² score: {train_model_score}, Test R² score: {test_model_score}")

        return report

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a pickled object from a file.
    
    Parameters:
    - file_path (str): The path to the pickled object.

    Returns:
    - object: The loaded object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        logging.error(f"Error loading object: {e}")
        raise CustomException(e, sys)
