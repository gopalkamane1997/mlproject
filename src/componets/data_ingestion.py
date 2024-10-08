import os
import sys
from src.exception import CustomException
from src.my_logging import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.componets.data_transform import dataTransformer
from src.componets.model_trainer import modelTrainerConfig, modelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class dataIngestion:
    def __init__(self):
        self.ingetstion_config = DataIngestionConfig()

    def initiate_data_ingetion(self):
        logging.info('Enterted the data inegetsion method or component')

        try:
            df = pd.read_csv('D:/data_science_projects/mlproject/notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingetstion_config.train_data_path),exist_ok=True)
            train_set,test_set = train_test_split(df,test_size = 0.2,random_state = 42)

            train_set.to_csv(self.ingetstion_config.train_data_path)
            test_set.to_csv(self.ingetstion_config.test_data_path)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingetstion_config.train_data_path,
                self.ingetstion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = dataIngestion()
    train_data, test_data = obj.initiate_data_ingetion()

    data_transformation  =  dataTransformer()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = modelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


