import re
import os
import sys
import time

from sklearn.model_selection import GridSearchCV
import dill
from src.exception import CustomException
from src.logger import logging

import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_high_corr_features(df, threshold=0.7):
    corr_matrix=df.corr()
    threshold = 0.70
    highly_correlated_features = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1,len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                highly_correlated_features.add(corr_matrix.columns[i])
                highly_correlated_features.add(corr_matrix.columns[j])
    return highly_correlated_features

# Define a custom transformer for the correlation step
class CorrelationTransformer:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.high_corr_features = None

    def transform(self, X):
        self.high_corr_features = get_high_corr_features(X, threshold=self.threshold)
        # Perform PCA on the subset of data
        subset = X[self.high_corr_features]
        pca = PCA(n_components=4)
        pca_result = pca.fit_transform(subset)
        # Create a new dataframe with the PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2','PC3','PC4'])
        # Concatenate the original dataframe with the PCA dataframe
        X_transformed = pd.concat([X, pca_df], axis=1)
        X_transformed = X_transformed.drop(columns=self.high_corr_features)
        return X_transformed

    def fit_transform(self, X, y=None):
        X_transformed = self.transform(X)
        logging.info(f"Correlation transform done")
        return X_transformed
    
# Define a custom transformer for the min-max scaling step
class MinMaxReverseTransformer:
    def __init__(self, columns=['age_of_car','age_of_policyholder']):
        self.columns = columns
        self.scaler = None

    def fit(self, X, y=None):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.inverse_transform(X_transformed[self.columns])
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X)
        X_transformed = self.transform(X)
        logging.info(f"MinMax transform done")
        return X_transformed
    
class StandardScale:
    def __init__(self,columns=None):
        self.columns=columns
        self.scaler = None

    def fit(self, X):
        self.scaler = StandardScaler()
        self.columns = list(X.select_dtypes(exclude='object').columns.values)
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        scaled_array = self.scaler.transform(X[self.columns])
        X_transformed = pd.DataFrame(data=scaled_array,columns=self.columns)
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X)
        logging.info("StandardScale fit_transform called from utils.py")
        return self.transform(X)

    
class OneHotEncode:
    def __init__(self,cols=None):
        self.cols=cols
        self.encoder = None

    def fit(self, X):
        self.encoder = OneHotEncoder(drop='first')
        self.cols=list(X.columns.values)
        self.encoder.fit(X[self.cols])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        encoded_array = self.encoder.transform(X[self.cols]).toarray()
        encoded_df = pd.DataFrame(data=encoded_array,columns=self.encoder.get_feature_names_out(self.cols))
        X_transformed = pd.concat([X_transformed,encoded_df],axis=1)
        X_transformed.drop(self.cols,axis=1,inplace=True)
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X)
        X_transformed = self.transform(X)
        logging.info(f"Onehot transform done")
        return X_transformed
    
def label_encoding_binary_features(df=None):
    try:
        pattern = re.compile(r'^is_')
        binary_features = [feature for feature in df.columns.values if pattern.match(feature)]
        for feature in binary_features:
            df[feature] = df[feature].map({'Yes':'1','No':'0'})
        logging.info(f"label encoder called from utils.py")
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
def combining_categorical_columns(df=None):
    try:
        max_rating,car_details,car_name=[],[],[]
        steering,locking,rear_window,parking,dashboard=[],[],[],[],[]
        for index,row in df.iterrows():
            car_details.append(f"{row['engine_type']} {row['rear_brakes_type']} {row['transmission_type']} {row['steering_type']} {row['fuel_type']}")
            car_name.append(f"{row['segment']} {row['model']}")
            max_rating.append(f"{row['max_torque']} {row['max_power']}")
            steering.append(str(int(row['is_adjustable_steering'])+int(row['is_power_steering'])))
            locking.append(str(int(row['is_power_door_locks'])+int(row['is_central_locking'])))
            rear_window.append(str(int(row['is_rear_window_wiper'])+int(row['is_rear_window_washer'])+int(row['is_rear_window_defogger'])))
            parking.append(str(int(row['is_parking_sensors'])+int(row['is_parking_camera'])))
            dashboard.append(str(int(row['is_tpms'])+int(row['is_ecw'])+int(row['is_speed_alert'])))
        
        columns_to_drop = ['engine_type','rear_brakes_type','transmission_type','steering_type','model','segment','max_power','max_torque','fuel_type','is_adjustable_steering','is_power_steering','is_power_door_locks','is_central_locking','is_rear_window_wiper','is_rear_window_washer','is_rear_window_defogger','is_parking_sensors','is_parking_camera','is_tpms','is_ecw','is_speed_alert']

        df.drop(columns_to_drop,axis=1,inplace=True)

        df['max_rating'] = max_rating
        df['car_details'] = car_details
        df['car_name'] = car_name
        df['steering'] = steering
        df['locking'] = locking
        df['rear_window'] = rear_window
        df['parking'] = parking
        df['dashboard'] = dashboard

        df.drop('policy_id',axis=1,inplace=True)

        logging.info(f"combined_categorical_columns called from utils.py")
        return df
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models, params=None):
    try:
        model_list=[]

        for key, value in models.items():
            
            logging.info(f"{key} training started")
            model = value
            # parameters = params[key]
            # grid_search = GridSearchCV(model,parameters,cv=5,n_jobs=-1)
            # grid_search.fit(X_train, y_train)
            # model.set_params(**grid_search.best_params_)
            start_time = time.time()
            model.fit(X_train, y_train)
            end_time = time.time()
    
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            roc = roc_auc_score(y_test,y_pred)
            elapsed_time = end_time-start_time

            model_list.append([key,accuracy,f1,roc,elapsed_time])
            logging.info(f"{key} training ended")
        
        return model_list
    except Exception as e:
        raise CustomException(e,sys)

    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)