from src.MLProject.components import data_ingestion
from src.MLProject.exception import CustomException
from src.MLProject.logger import logging
from src.MLProject.components.data_transformation import *
from src.MLProject.components.model_tranier import *
import sys
import mlflow
import dagshub
dagshub.init(repo_owner='Darkbeast747474', repo_name='MLProject', mlflow=True)

if __name__ == "__main__":
    logging.info("Executing App.py......")

    try:
        Data_Ingestion = data_ingestion.DataIngestion()
        train_data_path, test_data_path = Data_Ingestion.initiate_data_ingestion()

        Data_transformation = DataTransformation()
        train_arr, test_arr = Data_transformation.initiate_data_transormation(train_data_path, test_data_path)

        ModelTrainer = modelTrainer()
        print('r2_score',ModelTrainer.initiate_training(train_arr,test_arr))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
