import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train,X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]


            #Check if model has parameters defined
            if model_name in params:
                param_grid = params[model_name]
                print(f"Performing GridSearchCV for {model_name} with params: {param_grid}")

                # Perform grid search
                gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
                gs.fit(X_train, y_train)
                # Update model with best parameters
                model.set_params(**gs.best_params_)
                print(f"Best params for {model_name}: {gs.best_params_}")
            else:
                print(f"No parameters defined for {model_name}, using default parameters")
            # Fit model
            #model.fit(X_train, y_train)

            #Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Log the scores
            print(f"{model_name} - Train Score: {train_model_score:.4f}, Test Score: {test_model_score:.4f}")

            report[model_name] = test_model_score
            #report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
