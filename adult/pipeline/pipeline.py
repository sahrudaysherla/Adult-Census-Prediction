import uuid
import pandas as pd
import numpy as np
from adult.util.util import read_yaml_file, load_data, save_numpy_array_data, save_object
from adult.config import configuration
from adult.constant import *
from adult.components.data_ingestion import data_ingestion_component
from adult.components.data_validation import data_validation_component
from adult.components.data_transformation import data_transformation_component
from adult.components.model_trainer import model_trainer
from adult.components.model_evaluation import model_evaluation
from adult.components.model_pusher import ModelPusher
from adult.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelEvaluationArtifact, ModelPusherArtifact, ModelTrainerArtifact
from adult.logger import logging
from adult.exception import AdultException
from adult.entity.experiment import Experiment
from datetime import datetime
from collections import namedtuple
from threading import Thread
import sys, os

Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp", "artifact_time_stamp",
                                       "running_status", "start_time", "stop_time", "execution_time", "message",
                                       "experiment_file_path", "accuracy", "is_model_accepted"])

class pipeline():
    # experiment: Experiment = Experiment(*([None] * 11))
    # experiment_file_path = None
    def __init__(self, config: configuration) -> None:
        try:
            # os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            # pipeline.experiment_file_path=os.path.join(config.training_pipeline_config.artifact_dir,EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            # super().__init__(daemon=False, name="pipeline")
            self.pipeline_config = config
        except Exception as e:
            raise AdultException(e,sys) from e
        
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            logging.info(f"starting data ingestion at pipeline level")
            data_ingestion = data_ingestion_component(data_ingestion_config=self.pipeline_config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise AdultException(e,sys) from e

    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            logging.info(f"starting data validation at pipeline level")
            data_validation = data_validation_component(data_ingestion_artifact=data_ingestion_artifact,
                                                        data_validation_config=self.pipeline_config.get_data_validation_config())
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise AdultException(e,sys) from e

    def start_data_transformation(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_artifact: DataValidationArtifact)->DataTransformationArtifact:
        try:
            logging.info(f"starting data transformation at pipeline level")
            data_transformation = data_transformation_component(data_ingestion_artifact=data_ingestion_artifact,
                                                                data_validation_artifact=data_validation_artifact,
                                                                data_transformation_config=self.pipeline_config.get_data_transformation_config())
            return data_transformation.initiate_data_transformation()

        except Exception as e:
            raise AdultException(e,sys) from e

    def start_model_training(self, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            model_training = model_trainer(data_transformation_artifact=self.data_transformation_artifact,
                                            model_trainer_config=self.pipeline_config.get_model_training_config())
            return model_training.initiate_model_trainer()
        except Exception as e:
            raise AdultException(e,sys) from e
#we take them as inputs because attributes inside them are needed to call the evaluation function and it is likewise for all the function inputs in this project
    def start_model_evaluation(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_artifact: DataValidationArtifact, model_trainer_artifact:ModelTrainerArtifact)->ModelEvaluationArtifact: 
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact

            ModelEvaluation = model_evaluation(data_ingestion_artifact=self.data_ingestion_artifact,
                                                data_validation_artifact=self.data_validation_artifact,
                                                model_trainer_artifact=self.model_trainer_artifact,
                                                model_evaluation_config=self.pipeline_config.get_model_evaluation_config)
            return ModelEvaluation
        except Exception as e:
            raise AdultException(e,sys) from e

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            model_pusher = ModelPusher(model_pusher_config=self.pipeline_config.get_model_pusher_config(),
                                        model_evaluation_artifact=self.model_evaluation_artifact)
            return model_pusher
        except Exception as e:
            raise AdultException(e,sys) from e

        

    def run_pipeline(self):
        try:
            # if pipeline.experiment.running_status:
            #     logging.info("pipeline is already running")
            #     return pipeline.experiment
            # # data ingestion
            # logging.info("pipeline starting.")

            # experiment_id = str(uuid.uuid4())

            # pipeline.experiment = Experiment(experiment_id=experiment_id,
            #                                  initialization_timestamp=self.pipeline_config.current_time_stamp,
            #                                  artifact_time_stamp=self.pipeline_config.current_time_stamp,
            #                                  running_status=True,
            #                                  start_time=datetime.now(),
            #                                  stop_time=None,
            #                                  execution_time=None,
            #                                  experiment_file_path=pipeline.experiment_file_path,
            #                                  is_model_accepted=None,
            #                                  message="pipeline has been started.",
            #                                  accuracy=None,
            #                                  )
            # logging.info(f"pipeline experiment: {pipeline.experiment}")

            # self.save_experiment()
            #-----------------
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                        data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_training(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_validation_artifact=data_validation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact,)
        except Exception as e:
            raise AdultException(e,sys) from e    
        #---------------------------------
        #     if model_evaluation_artifact.is_model_accepted:
        #         model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
        #         logging.info(f'Model pusher artifact: {model_pusher_artifact}')
        #     else:
        #         logging.info("Trained model rejected.")
        #     logging.info("pipeline completed.")

        #     stop_time = datetime.now()
        #     pipeline.experiment = Experiment(experiment_id=pipeline.experiment.experiment_id,
        #                                      initialization_timestamp=self.pipeline_config.current_time_stamp,
        #                                      artifact_time_stamp=self.pipeline_config.current_time_stamp,
        #                                      running_status=False,
        #                                      start_time=pipeline.experiment.start_time,
        #                                      stop_time=stop_time,
        #                                      execution_time=stop_time - pipeline.experiment.start_time,
        #                                      message="pipeline has been completed.",
        #                                      experiment_file_path=pipeline.experiment_file_path,
        #                                      is_model_accepted=model_evaluation_artifact.is_model_accepted,
        #                                      accuracy=model_trainer_artifact.model_accuracy
        #                                      )
        #     logging.info(f"pipeline experiment: {pipeline.experiment}")
        #     self.save_experiment()

        # except Exception as e:
        #     raise AdultException(e,sys) from e

    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise AdultException(e,sys) from e
    # def save_experiment(self):
    #     try:
    #         if pipeline.experiment.experiment_id is not None:
    #             experiment = pipeline.experiment
    #             experiment_dict = experiment._asdict()
    #             experiment_dict: dict = {key: [value] for key, value in experiment_dict.items()}

    #             experiment_dict.update({
    #                 "created_time_stamp": [datetime.now()],
    #                 "experiment_file_path": [os.path.basename(pipeline.experiment.experiment_file_path)]})

    #             experiment_report = pd.DataFrame(experiment_dict)

    #             os.makedirs(os.path.dirname(pipeline.experiment_file_path), exist_ok=True)
    #             if os.path.exists(pipeline.experiment_file_path):
    #                 experiment_report.to_csv(pipeline.experiment_file_path, index=False, header=False, mode="a")
    #             else:
    #                 experiment_report.to_csv(pipeline.experiment_file_path, mode="w", index=False, header=True)
    #         else:
    #             print("First start experiment")
    #     except Exception as e:
    #         raise AdultException(e, sys) from e

    # @classmethod
    # def get_experiments_status(cls, limit: int = 5) -> pd.DataFrame:
    #     try:
    #         if os.path.exists(pipeline.experiment_file_path):
    #             df = pd.read_csv(pipeline.experiment_file_path)
    #             limit = -1 * int(limit)
    #             return df[limit:].drop(columns=["experiment_file_path", "initialization_timestamp"], axis=1)
    #         else:
    #             return pd.DataFrame()
    #     except Exception as e:
    #         raise AdultException(e, sys) from e

        