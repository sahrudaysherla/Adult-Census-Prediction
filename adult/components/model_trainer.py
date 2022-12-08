from adult.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact
from adult.entity.config_entity import ModelTrainerConfig
from adult.entity.model_factory import model_factory, MetricInfoArtifact, GridSearchedBestModel, evaluate_regression
import numpy as np
import pandas as pd
from adult.constant import *
from adult.util.util import read_yaml_file, load_data, save_numpy_array_data, save_object, load_np_data, load_object
from adult.logger import logging
from adult.exception import AdultException
from typing import List
import os,sys

class Estimator:
    def __init__(self, preprocessing_object, trained_model_object) -> None:
        self.preprocessing_object = preprocessing_object
        self.trained_object = trained_model_object

    def predict(self, X_data):
        transformed_features = self.preprocessing_object.transform(X_data)
        return self.trained_object.predict(transformed_features)



class model_trainer:
    def __init__(self, data_transformation_artifact = DataTransformationArtifact,model_trainer_config = ModelTrainerConfig) -> None:
        try:
            logging.info(f"Model trainer component entered")
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config

            
        except Exception as e:
            raise AdultException(e,sys) from e

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Loading Transformed Data")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_data_path
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_data_path
            train_data = load_np_data(transformed_train_file_path)
            test_data = load_np_data(transformed_test_file_path)
            Xtrain, yTrain, Xtest, yTest = train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

            logging.info(f"Extracting model information")
            model_config_info = self.model_trainer_config.model_config_path
            
            logging.info(f"Starting Model Factory on the {model_config_info}")
            ModelFactory = model_factory(model_config_path=model_config_info)

            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"base accuracy is {base_accuracy}")

            logging.info(f"Model Selection operation started")
            best_model = ModelFactory.get_best_model(Xtrain, yTrain, base_accuracy=base_accuracy)

            logging.info(f"Best Model is {best_model}")

            logging.info(f"Extracting trained model list")
            grid_search_best_model_list: List[GridSearchedBestModel] = ModelFactory.grid_searched_best_model_list
            model_list = [model.best_model for model in grid_search_best_model_list]
            logging.info(f"Evaluating model on training and testing dataset")
            metric_info:MetricInfoArtifact = evaluate_regression(model_list=model_list,
                                                                Xtrain=Xtrain,
                                                                Xtest=Xtest,
                                                                yTrain=yTrain,
                                                                yTest=yTest,
                                                                base_accuracy=base_accuracy)
            logging.info(f"Best Model Found: {metric_info}")

            preprocessing_obj = load_object(self.data_transformation_artifact.preprocessed_file_path)

            model_object = metric_info.model_object

            trained_model_file_path = self.model_trainer_config.trained_model_path
            adult_model = Estimator(preprocessing_object=preprocessing_obj, trained_model_object=model_object)
            logging.info(f"Saving model at {trained_model_file_path}")
            save_object(trained_model_file_path, adult_model)


            model_trainer_artifact = ModelTrainerArtifact(is_trained=True, message=f"Model has been trained successfully",
                                                        trained_model_file_path=trained_model_file_path,
                                                        train_accuracy=metric_info.train_accuracy,
                                                        test_accuracy=metric_info.test_accuracy,
                                                        model_accuracy=metric_info.model_accuracy,
                                                        test_rmse=metric_info.test_rmse,
                                                        train_rmse=metric_info.train_rmse)
            return model_trainer_artifact
        except Exception as e:
            raise AdultException(e,sys) from e

    def __delattr__(self, __name: str) -> None:
        logging.info(f"Model Training Completed")




        