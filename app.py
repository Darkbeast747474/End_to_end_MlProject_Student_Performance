from src.MLProject.components import data_ingestion
from src.MLProject.exception import CustomException
from src.MLProject.logger import logging
import sys

if __name__ == '__main__':
    logging.info('Executing App.py......')
    
    try:
        Data_Ingestion = data_ingestion.DataIngestion()
        Data_Ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info('Custom Exception')        
        raise CustomException(e,sys)