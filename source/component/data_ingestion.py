from source.logger import logging
from source.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
import sys
import pandas as pd



# diffing variable so use data class

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifact','train.csv')
    test_data_path = os.path.join('artifact','test.csv')
    raw_data_path = os.path.join('artifact','data.csv')

class DataIngestion:
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # ingestion_config storing path of train, test, raw data
    
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('notebook/data/UCI_Credit_Card.csv')
            logging.info('read the data set as datafram')

            #creating folder where we can save training data file
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # saving the raw data
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            # splitting the data into train and test
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df,test_size=0.25, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Train test split completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()