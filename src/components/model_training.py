from dataclasses import dataclass
import os
import sys
import re
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_data):
        try:
            logging.info("Model Training initiated")
            logging.info("Train-validation-test split")
            X_train, X_test, y_train, y_test = train_test_split(train_data.drop('target',axis=1),train_data['target'],
                                                                test_size=0.25,shuffle=True,stratify=train_data['target'],random_state=24)
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(),
                "AdaBoost": AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1)),
                "CatBoost": CatBoostClassifier(),
                "GradientBoost": GradientBoostingClassifier()
            }

            '''params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10, 100]
                },
                "K-Neighbors": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"]
                },
                "Decision Tree": {
                    "max_depth": [3, 5, 7, 9, 11],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                #"Random Forest": {
                    #"n_estimators": [100, 250, 500],
                    #"max_depth": [3, 5, 7, 9, 11],
                    #"min_samples_split": [2, 5, 10],
                    #"min_samples_leaf": [1, 2, 4]
                #},
                "XGBoost": {
                    "n_estimators": [100, 250, 400, 600],
                    "max_depth": [3, 5, 7, 9, 11],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5, 1],
                    "subsample": [0.5, 0.75, 1],
                    "colsample_bytree": [0.5, 0.75, 1],
                    "gamma": [0, 0.1, 1],
                    "reg_alpha": [0, 0.1, 1],
                    "reg_lambda": [0, 0.1, 1]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 250],
                    "learning_rate": [0.01, 0.1, 1],
                },
                "CatBoost": {
                    "iterations": [100, 250, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5, 1],
                    "depth": [3, 5, 7, 9, 11],
                    "l2_leaf_reg": [1, 3, 5]
                },
                "GradientBoost": {
                    "n_estimators": [100, 250, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5, 1],
                    "max_depth": [3, 5, 7, 9, 11],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.5, 0.75, 1],
                    "max_features": ["sqrt", "log2"]
                }
            }'''

            model_report:list = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info("Model Testing Completed")
            
            report_df = pd.DataFrame(data=model_report,columns=['Model name','Accuracy','f1-score','ROC-AUC score','Time (in seconds)'])
            logging.info("Choosing the best model")
            min_time=500
            best_model=""
            for index,row in report_df.iterrows():
                if row['Accuracy']>0.95 and row['f1-score']>0.95:
                    if row['Time (in seconds)']<min_time:
                        min_time=row['Time (in seconds)']
                        best_model=row['Model name']
            logging.info(f"Best Model is: {best_model}")
            logging.info(f"Training the best model")
            model = models[best_model]
            #model.set_params(report_df[report_df['Model name']==best_model]['best parameters'])
            model.fit(X_train,y_train)

            logging.info("Saving the best model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            return(report_df[report_df['Model name']==best_model]['Model name'], 
                report_df[report_df['Model name']==best_model]['Accuracy']
            )

        except Exception as e:
            raise CustomException(e,sys)




