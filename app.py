from src.MLProject.components import data_ingestion
from src.MLProject.exception import CustomException
from src.MLProject.logger import logging
from src.MLProject.components.data_transformation import *
import sys

if __name__ == "__main__":
    logging.info("Executing App.py......")

    try:
        Data_Ingestion = data_ingestion.DataIngestion()
        train_data_path, test_data_path = Data_Ingestion.initiate_data_ingestion()

        Data_transformation = DataTransformation()
        Data_transformation.initiate_data_transormation(train_data_path, test_data_path)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
