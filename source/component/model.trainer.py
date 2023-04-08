import pandas as pd
import numpy as np
import os
import sys
from source.logger import logging
from source.exception import CustomException
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from source.utils import save_object, evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_mode_trainer(self,train_array, test_array):
        try:
            logging.info('split training and test input data')
            x_train,y_train,x_test,y_test =(
                train_array[:,:-1]
                ,train_array[:,-1], test_array[:,:-1],test_array[:,-1]
            )

            models = {
                'Random Forest': RandomForestClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'XGBClassifier': XGBClassifier(),
                'CatBoosting': CatBoostClassifier(),
                'Logistic Regression': LogisticRegression(),
                'AdaBoost Classifier': AdaBoostClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'K-Neighbors Classifier': KNeighborsClassifier()
            }

            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test)

            # to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name = list(model_report.values())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best Model found')
            logging.info('Best found model on both training and test dataset is {best_model}')

            save_object(
                self.model_trainer_config.trained_model_file_path, best_model,
                obj=best_model)
            
            predicted = best_model.predict(x_test)
            roc_auc= roc_auc_score(x_test, predicted)
            return roc_auc
                                    
        except Exception as e:
            raise CustomException(e,sys)
        