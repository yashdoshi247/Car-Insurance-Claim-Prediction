import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder,StandardScaler
from imblearn.over_sampling import ADASYN

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import CorrelationTransformer, MinMaxReverseTransformer, OneHotEncode, StandardScale, combining_categorical_columns, label_encoding_binary_features, save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,df):

        try:
            numerical_columns = list(df.select_dtypes(exclude='object').columns.values)
            categorical_columns = list(df.select_dtypes(include='object').columns.values)
            
            logging.info("Creating numerical pipeline")
            num_pipeline=Pipeline(
                steps=[
                    ('correlation', CorrelationTransformer()),
                    ('minmax_scaling', MinMaxReverseTransformer()),
                    ('standard_scaling', StandardScale())
                ]
            )

            logging.info("Creating categorical pipeline")
            cat_pipeline = Pipeline(
                steps=[
                    ('label_encoding_binary_features', FunctionTransformer(label_encoding_binary_features)),
                    ('combining_categorical_columns', FunctionTransformer(combining_categorical_columns)),
                    ('One_hot_encoding', OneHotEncode())
                ]
            )

            logging.info("Creating preprocessor object")
            preprocessor = ColumnTransformer(
                [
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,data_path):

        try:
            df = pd.read_csv(data_path)

            logging.info("Reading Dataset completed")
            logging.info("Obtaining preprocessing object")

            logging.info("Transforming training data")
            X = df.drop('is_claim',axis=1)
            y = df['is_claim']
            preprocessing_obj = self.get_data_transformer_object(X)
            X_transformed = preprocessing_obj.fit_transform(X)

            logging.info(f"Oversampling training data")
            adasyn = ADASYN()
            X_resampled, y_resampled = adasyn.fit_resample(X_transformed, y.values)
            X_resampled_df = pd.DataFrame(X_resampled, columns=[f"feature_{i}" for i in range(X_resampled.shape[1])])
            y_resampled_df = pd.DataFrame(y_resampled, columns=["target"])

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                pd.concat([X_resampled_df,y_resampled_df],axis=1),
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
