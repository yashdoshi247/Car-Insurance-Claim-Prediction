import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_preprocessing import DataTransformation
from src.components.data_preprocessing import DataTransformationConfig
import pandas as pd

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")

class DataIngestion:

    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            train_df=pd.read_csv('train.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")

            return self.ingestion_config.train_data_path
        except Exception as e:
            raise CustomException(e,sys)
            