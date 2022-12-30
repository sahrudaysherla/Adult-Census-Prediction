import pandas as pd
from adult.exception import AdultException
from adult.logger import logging
from adult.constant import *
from adult.util.util import read_yaml_file
import os, sys
from collections import namedtuple
from sklearn.metrics import r2_score, mean_squared_error, precision_score, accuracy_score
import importlib
from pyexpat import model
from typing import List
import numpy as np
import yaml
GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"

InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score",
                                                             ])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score", ])

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])

def evaluate_classification(model_list:List, Xtrain:np.array, Xtest:np.array, yTrain:np.array, yTest:np.array, base_accuracy:float=0.6)->MetricInfoArtifact:
    try:
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
                logging.info(f"{'>'*10}Starting Evaluation of the model {str(model)} {'<'*10}")
                y_train_pred = model.predict(Xtrain)
                y_test_pred = model.predict(Xtest)
                
                logging.info(f"Calculating accuracy of the model")

                train_accuracy =  accuracy_score(yTrain, y_train_pred)
                test_accuracy = accuracy_score(yTest, y_test_pred)

                logging.info(f"Train accuracy is {train_accuracy} and Test accuracy is {test_accuracy} of model {str(model)}")
                logging.info(f"Calculating Error")

                train_rmse = 1 - train_accuracy
                test_rmse = 1 - test_accuracy

                logging.info(f"Train error rate is {train_rmse} and Test error rate is {test_rmse} of model {str(model)}")

                model_accuracy = (2*(train_accuracy*test_accuracy))/(train_accuracy+test_accuracy)
                diff_in_accuracy = abs(test_accuracy - train_accuracy)

                logging.info(f"model accuracy is {model_accuracy} and difference is {diff_in_accuracy}")
                model_name = str(model)

                if model_accuracy >= base_accuracy and diff_in_accuracy <= float(0.05):
                    base_accuracy = model_accuracy
                    metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                    model_object=model,
                                                    train_rmse=train_rmse,
                                                    test_rmse=test_rmse,
                                                    train_accuracy=test_accuracy,
                                                    test_accuracy=test_accuracy,
                                                    index_number=index_number,
                                                    model_accuracy=model_accuracy)

                    logging.info(f"Acceptable model found {metric_info_artifact}. ")
                index_number += 1
        if metric_info_artifact is None:
            logging.info(f"Metric Info Artifact is {metric_info_artifact}")
            logging.info(f"No model found with higher accuracy than base accuracy or prev model")
        logging.info(f"Metric Info Artifact is {metric_info_artifact}")
        return metric_info_artifact
    except Exception as e:
        raise AdultException(e,sys) from e


def evaluate_regression(model_list:List, Xtrain:np.array, Xtest:np.array, yTrain:np.array, yTest:np.array, base_accuracy:float=0.6)->MetricInfoArtifact:
    try:
        index_number = 0
        
        for model in model_list:
            logging.info(f"{'>'*10}Starting Evaluation of the model {str(model)} {'<'*10}")
            y_train_pred = model.predict(Xtrain)
            y_test_pred = model.predict(Xtest)
            
            logging.info(f"Calculating accuracy of the model")

            train_accuracy =  r2_score(yTrain, y_train_pred)
            test_accuracy = r2_score(yTest, y_test_pred)

            logging.info(f"Train accuracy is {train_accuracy} and Test accuracy is {test_accuracy} of model {str(model)}")
            logging.info(f"Calculating RMSE")

            train_rmse = mean_squared_error(yTrain,y_train_pred)
            test_rmse = mean_squared_error(yTest, y_test_pred)

            logging.info(f"Train rmse is {train_rmse} and Test rmse is {test_rmse} of model {str(model)}")

            model_accuracy = (2*(train_accuracy*test_accuracy))/(train_accuracy+test_accuracy)
            diff_in_accuracy = abs(test_accuracy - train_accuracy)

            logging.info(f"model accuracy is {model_accuracy} and difference is {diff_in_accuracy}")
            model_name = str(model)

            # if model_accuracy >= base_accuracy and diff_in_accuracy <= 0.05:
            #     base_accuracy = model_accuracy
            metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                            model_object=model,
                                            train_rmse=train_rmse,
                                            test_rmse=test_rmse,
                                            train_accuracy=test_accuracy,
                                            test_accuracy=test_accuracy,
                                            index_number=index_number,
                                            model_accuracy=model_accuracy)

            logging.info(f"Acceptable model found {metric_info_artifact}. ")
            index_number += 1
            # if metric_info_artifact is None:
            #     logging.info(f"No model found with higher accuracy than base accuracy")
            return metric_info_artifact
    except Exception as e:
        raise AdultException(e,sys) from e












class model_factory:
    def __init__(self, model_config_path:str = None) -> None:
        self.model_config = self.read_params(model_config_path)
        self.initialized_model_config: dict = dict(self.model_config[MODEL_SELECTION_KEY])
        self.grid_search_class = self.model_config[GRID_SEARCH_KEY][CLASS_KEY]
        self.grid_seach_module = self.model_config[GRID_SEARCH_KEY][MODULE_KEY]
        self.grid_search_params = self.model_config[GRID_SEARCH_KEY][PARAM_KEY]

    @staticmethod
    def read_params(config_path:str)-> dict:
        try:
            logging.info(f"Entering read_params")
            with open(config_path) as file:
                config_attributes = yaml.safe_load(file)

            return config_attributes
        except Exception as e:
            raise AdultException(e, sys) from e

    @staticmethod
    def update_property_class(instance_reference: object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception(f"Please pass data in the form of dict")
            print(property_data)
            logging.info(f"{property_data}")
            for key, value in property_data.items():
                setattr(instance_reference, key, value) #this sets the key attribute of the instance object to the given value while getattr(module, attribute) gets the value of the passed attribute of the passed object


            return instance_reference
        except Exception as e:
            raise AdultException(e, sys) from e
    @staticmethod
    def import_class_from_module(module_name:str, class_name:str):
        try:
            module = importlib.import_module(module_name)
            logging.info(f"Importing {class_name} from {module_name}")
            get_class = getattr(module, class_name)
            return get_class
        except Exception as e:
            raise AdultException(e, sys) from e

    def get_initialized_model_list(self)->List[InitializedModelDetail]:
        try:
            initialized_model_list = []
            logging.info(f"Entering get_initialized_model_list")

            for model_serial_number in self.initialized_model_config.keys():
                model_initialization_config = self.initialized_model_config[model_serial_number]
                model_module = model_initialization_config[MODULE_KEY]
                model_class = model_initialization_config[CLASS_KEY]
                model_import = self.import_class_from_module(model_module, model_class)
                model_obj = model_import()
                if PARAM_KEY in model_initialization_config:
                    logging.info(f"Accessing and updating Params")
                    model_obj_prop_data = dict(model_initialization_config[PARAM_KEY])
                    model_obj = model_factory.update_property_class(model_obj,model_obj_prop_data)

                params_search_grid = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_module}.{model_class}"
                model_initialization_config = InitializedModelDetail(model_serial_number=model_initialization_config,
                                                                    model=model_obj, param_grid_search=params_search_grid, model_name=model_name)

                initialized_model_list.append(model_initialization_config)
            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise AdultException(e, sys) from e
        
    def execute_grid_search(self, initialized_model_detail: InitializedModelDetail, input_feature, output_feature)-> GridSearchedBestModel:
        try:

            grid_search_ref = model_factory.import_class_from_module(module_name=self.grid_seach_module, class_name=self.grid_search_class)
            grid_search_obj = grid_search_ref(estimator = initialized_model_detail.model, param_grid = initialized_model_detail.param_grid_search)
            grid_search_obj = model_factory.update_property_class(grid_search_obj, self.grid_search_params)
            logging.info(f"{'>>'*30}Training {type(initialized_model_detail.model).__name__} Started.{'<<'*30}")

            grid_search_obj.fit(input_feature, output_feature)
            logging.info(f"{grid_search_obj}")
            logging.info(f"{'>>'*30}Training {type(initialized_model_detail.model).__name__} Completed.{'<<'*30}")
            return GridSearchedBestModel(
                model_serial_number=initialized_model_detail.model_serial_number,
                model=model,
                best_model=grid_search_obj.best_estimator_,
                best_parameters=grid_search_obj.best_params_,
                best_score=grid_search_obj.best_score_
            )

        except Exception as e:
            raise AdultException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            logging.info(f"Initiating best parameter search for initialized model")
            return self.execute_grid_search(initialized_model_detail=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise AdultException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:

        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_list,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise AdultException(e, sys) from e

    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:
        """
        This function return ModelDetail
        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise AdultException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy=0.5
                                                          ) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found:{grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise AdultException(e, sys) from e

    def get_best_model(self, X, y,base_accuracy=0.5) -> BestModel:
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            return model_factory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_accuracy=base_accuracy)
        except Exception as e:
            raise AdultException(e, sys) from e



        

