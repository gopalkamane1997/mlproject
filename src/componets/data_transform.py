import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.my_logging import logging

import os

from src.utils import save_object #for saving pkl file

@dataclass
class dataTransformerConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class dataTransformer:
    def __init__(self):
        self.data_transformation_config = dataTransformerConfig()
    
    def get_data_transformer_object(self):

        '''
        This function is reponsible for data transformation
        '''

        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipelone = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy = 'most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'numerical columns{numerical_columns}')
            logging.info(f'categoriacal columns {categorical_columns}')

            preprocessing = ColumnTransformer([
                ('num_pipeline',num_pipelone,numerical_columns),
                ('categorical_pipeline',categorical_pipeline,categorical_columns)
            ])
            return preprocessing
    
        except Exception as e:
            raise CustomException(e,sys) 
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('reading train and test data completed')

            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'math_score'
            numerical_columns = ['writing_score','reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('applying preprocessor object on training dataframe and testing dataframe')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info('saving preprocessing objects')
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj  = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)





        