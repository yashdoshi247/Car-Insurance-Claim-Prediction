from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataTransformation
from src.components.model_training import ModelTrainer
from src.logger import logging
import sys
print(sys.path)

data_ingestion = DataIngestion()
train_data_path = data_ingestion.initiate_data_ingestion()
logging.info("Data Ingestion Completed")

data_preprocessing = DataTransformation()
train_data_preprocessed,_ = data_preprocessing.initiate_data_transformation(train_data_path)
logging.info("Data preprocessing completed")

model_training = ModelTrainer()
best_model, best_model_accuracy = model_training.initiate_model_trainer(train_data_preprocessed)
logging.info("Model training completed")
logging.info(f"Best model: {best_model}, Accuracy: {best_model_accuracy}")
