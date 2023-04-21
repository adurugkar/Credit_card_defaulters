from source.logger import logging
from source.exception import CustomException
from source.component.data_transformation import DataTransformation
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
import sys
import pandas as pd
from source.component.model_trainer import ModelTrainer
from imblearn.over_sampling import SMOTE


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
            
            # Data set column rename
            logging.info(f"columns name defore renaming the colomns : {df.columns}")
            df.rename(columns={'PAY_0':'PAY_SEPT','PAY_2':'PAY_AUG','PAY_3':'PAY_JUL','PAY_4':'PAY_JUN','PAY_5':'PAY_MAY','PAY_6':'PAY_APR','default.payment.next.month':'DEFAULT'},inplace=True)
            df.rename(columns={'BILL_AMT1':'BILL_AMT_SEPT','BILL_AMT2':'BILL_AMT_AUG','BILL_AMT3':'BILL_AMT_JUL','BILL_AMT4':'BILL_AMT_JUN','BILL_AMT5':'BILL_AMT_MAY','BILL_AMT6':'BILL_AMT_APR'}, inplace = True)
            df.rename(columns={'PAY_AMT1':'PAY_AMT_SEPT','PAY_AMT2':'PAY_AMT_AUG','PAY_AMT3':'PAY_AMT_JUL','PAY_AMT4':'PAY_AMT_JUN','PAY_AMT5':'PAY_AMT_MAY','PAY_AMT6':'PAY_AMT_APR'},inplace=True)
            
            logging.info(f"columns name after renaming {df.columns}")

           
            logging.info(f"balancing in Education and marriage")
            df.replace({'EDUCTION':{1:1,2:1,3:2,4:3,5:3,6:3,0:3},'MARRIAGE':{1:1,2:2,0:3,3:3}},inplace=True)

            ind_data = df.iloc[:,:-1] # independent_feature
            dep_data = df.iloc[:,-1] # dependent_feature

            # data balancing using smote
            logging.info('data set is imbalance to balancing it using smote')
            smote = SMOTE(sampling_strategy='minority')
            x_sm, y_sm = smote.fit_resample(ind_data, dep_data)

            #data concatenation
            data = pd.merge(x_sm, y_sm, left_index=True, right_index=True)


            # splitting the data into train and test

            train_set, test_set = train_test_split(data,test_size=0.25, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print('ingestion done')

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    print('data transformation done')

    modeltrainer = ModelTrainer()
    auc_roc_socres = modeltrainer.initiate_mode_trainer(train_arr,test_arr)
    print('model training get done')
    print(auc_roc_socres)