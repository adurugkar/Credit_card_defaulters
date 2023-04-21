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
            x_train,y_train,x_test,y_test =(train_array[:,:-1],train_array[:,-1], test_array[:,:-1],test_array[:,-1])

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
            
            params = {
                'Random Forest':{
                    'n_estimators':[800,1000],
                    # 'max_depth':[4,5,6],
                    # 'min_samples_split':[2,4,5],
                    # 'max_features' = ['sqrt', 'log2']
                    # 'criterion':['gini','entropy']
                },
                'Decision Tree':{
                    
                    'max_depth':[4,5,6],
                    # 'min_samples_split':[2,4,5],
                    # 'max_features' : ['sqrt', 'log2']
                    # 'criterion':['gini','entropy']                
                },
                'XGBClassifier':{
                    'learning_rate':[0.01,0.001],
                    # 'n_estimators':[600,800],
                    # 'max_depth':[4,5,6],
                    # 'min_child_weight':[4,5,6],
                    # 'gamma':[i/10.0 for i in range(0,4)]
                },
                'CatBoosting':{
                    'n_estimators':[100,200,300],
                    # 'learning_rate':[0., 0.05, 0.1 , 0.15, 0.2 ],
                    # 'max_depth': [3,4,5],
                #     'rsm':[],
                #     'loss_function':[]
                },
                'Logistic Regression':{
                    'max_iter':[400,300],
                    # 'solver':['lbfgs','sag','saga','liblinear'],
                    # 'c_values':[100,10,1,0.1,0.01]
                },
                'AdaBoost Classifier':{
                    'n_estimators': [2, 3, 4],
                    # 'learning_rate': [(0.97 + x / 100) for x in range(0, 4)],
                    # 'algorithm': ['SAMME', 'SAMME.R']
                },

                'Gradient Boosting':{
                    'n_estimators':[10, 100, 1000],
                    # 'learning_rate':[0.001, 0.01, 0.1],
                    # 'subsample':[0.5, 0.7, 1.0],
                    # 'max_depth':[3, 7, 9]
                },
                'K-Neighbors Classifier':{
                    'n_neighbors':range(1, 21, 2),
                    # 'weights' :['uniform', 'distance'],
                    # 'metric':['euclidean', 'manhattan', 'minkowski']
                }
                }
            logging.info('model evaluation get start')
            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models=models,params=params)
            logging.info(f'model report : {model_report}')

            # to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f'best model : {best_model_name}')

            if best_model_score<0.6:
                raise CustomException('No best Model found')
            logging.info(f'Best found model on both training and test dataset is {best_model}')

            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=best_model)
            logging.info(f'best model save in path : {self.model_trainer_config.trained_model_file_path}')
            
            logging.info('predicting the y_test value ')
            predicted = best_model.predict(x_test)
            roc_auc= roc_auc_score(y_test, predicted)
            logging.info(f'we get roc_auc_sorce : {roc_auc}')
            return roc_auc
                                    
        except Exception as e:
            raise CustomException(e,sys)
        