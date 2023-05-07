import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

from main_folder.exception import CustomException
from main_folder.logger import logging

from main_folder.utils import save_object,evaluate_model
import sys
import os

from dataclasses import dataclass

@dataclass
class ModalTrainConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModalTrainConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Aplitting Dependent and Independent variables from train and test')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                'LogisticRegression':LogisticRegression()
            }
            
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            
            print(model_report)
            logging.info(f'Model Report : {model_report}')
            
            #TO get best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            logging.info(f'Model Name : {best_model_name} , Accuracy_score : {best_model_score * 100}')
            print('\n==========================================================================')
            print(f'Model Name : {best_model_name} , Accuracy_score : {best_model_score * 100}')
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            
            
        except Exception as e:
            logging.info('Exception occured at the Model training')
            raise CustomException(e,sys)
          