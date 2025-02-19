import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10, 20],
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                },
                "Linear Regression": {},  # No hyperparameters to tune
                "XGBRegressor": {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1],
                },
                "CatBoosting Regressor": {},  # No hyperparameters to tune
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [1.0, 0.1],
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7],
                }
            }

            # Evaluate models
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # To get best model score from dict
            best_model_score = max(model_report.values())

            # To get best model name from dict
            best_model_name = [model for model, score in model_report.items() if score == best_model_score]

            if not best_model_name or best_model_score < 0.6:
                raise CustomException("No best model found")

            best_model = models[best_model_name[0]]  # Take the first best model

            logging.info(f"Best found model: {best_model_name[0]} with score: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predictions
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score of the best model: {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
