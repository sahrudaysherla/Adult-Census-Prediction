from adult.logger import logging
from adult.exception import AdultException
from adult.util.util import read_yaml_file
from adult.constant import *
from adult.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
import os,sys
class Configuration:
    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 config_time_stamp = CURRENT_TIME_STAMP) -> None:
                 self.config_info = read_yaml_file(config_file_path)
                 self.training_pipeline_config = self.get_training_pipeline_config()
                 self.current_time_stamp = config_time_stamp
        
    def get_data_ingestion_config(self)->DataIngestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_ingestion_artifact_dir = os.path.join(artifact_dir,DATA_INGESTION_ARTIFACT_DIR,self.current_time_stamp)

            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY]

            dataset_download_url = data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY]
            tgz_download_dir = os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY])
            raw_data_dir = os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY])
            ingested_dir = os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
            ingested_train_dir = os.path.join(ingested_dir,data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY])
            ingested_test_dir = os.path.join(ingested_dir,data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY])
            data_ingestion_config = DataIngestionConfig(dataset_download_url=dataset_download_url,
                                                        tgz_download_dir=tgz_download_dir,
                                                        raw_data_dir=raw_data_dir,
                                                        ingested_train_dir=ingested_train_dir,
                                                        ingested_test_dir=ingested_test_dir)
            return data_ingestion_config
        except Exception as e:
            raise AdultException(e,sys) from e

    def get_data_validation_config(self)->DataValidationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_validation_artifact_dir = os.path.join(artifact_dir, DATA_VALIDATION_ARTIFACT_DIR_KEY,self.current_time_stamp)
            data_validation_info = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            schema_file_path = os.path.join(ROOT_DIR,data_validation_info[DATA_VALIDATION_SCHEMA_DIR_KEY],data_validation_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])
            report_file_path = os.path.join(data_validation_artifact_dir,data_validation_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY])
            report_page_file_path = os.path.join(data_validation_artifact_dir,data_validation_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME])
            data_validation_config = DataValidationConfig(schema_file_path=schema_file_path,
                                                            report_file_path=report_file_path,
                                                            report_page_file_path=report_page_file_path)
            return data_validation_config

        except Exception as e:
            raise AdultException(e,sys) from e
            
    def get_data_transformation_config(self)-> DataTransformationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_transformation_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            data_transformation_artifact_dir = os.path.join(artifact_dir, DATA_TRANSFORMATION_ARTIFACT_DIR_KEY, self.current_time_stamp)
            transformed_train_path = os.path.join(data_transformation_artifact_dir, data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY],data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY])
            transformed_test_path = os.path.join(data_transformation_artifact_dir, data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY],data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY])
            preprocessed_file_path = os.path.join(data_transformation_artifact_dir,data_transformation_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],data_transformation_info[DATA_TRANSFORMATION_PREPROCESSED_OBJ_FILE_NAME_KEY])
            data_transformation_config = DataTransformationConfig(transformed_test_file_path=transformed_test_path,
                                                                transformed_train_file_path = transformed_train_path,
                                                                preprocessed_file_path=preprocessed_file_path)
            return data_transformation_config

        except Exception as e:
            raise AdultException(e,sys) from e

    def get_model_training_config(self)->ModelTrainerConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            model_training_info = self.config_info[MODEL_TRAINING_CONFIG_KEY]
            model_training_artifact_dir = os.path.join(artifact_dir, MODEL_TRAINING_ARTIFACT_DIR_KEY , get_curr_timestamp())
            trained_model_path = os.path.join(model_training_artifact_dir, model_training_info[MODEL_TRAINING_TRAINED_MODEL_DIR_KEY], model_training_info[MODEL_TRAINING_TRAINED_MODEL_FILE_NAME_KEY])
            model_config_path = os.path.join(ROOT_DIR, model_training_info[MODEL_TRAINING_BASE_CONF_DIR_KEY], model_training_info[MODEL_TRAINING_BASE_CONF_FILE_NAME_KEY])
            model_trainer_config = ModelTrainerConfig(base_accuracy=0.3,
                                                    trained_model_path=trained_model_path,
                                                    model_config_path=model_config_path)
            return model_trainer_config


        except Exception as e:
            raise AdultException(e,sys) from e

    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        try:
            model_evaluation_config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            artifact_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                        MODEL_EVALUATION_ARTIFACT_DIR, )

            model_evaluation_file_path = os.path.join(artifact_dir,
                                                    model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY])
            response = ModelEvaluationConfig(model_evaluation_file_path=model_evaluation_file_path,
                                            time_stamp=self.current_time_stamp)
            
            
            logging.info(f"Model Evaluation Config: {response}.")
            return response
        
        except Exception as e:
            raise AdultException(e,sys) from e

    def get_model_pusher_config(self)->ModelPusherConfig:
        try:
            time_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_pusher_config_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            export_dir_path = os.path.join(ROOT_DIR, model_pusher_config_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                           time_stamp)

            model_pusher_config = ModelPusherConfig(export_dir_path=export_dir_path)
            logging.info(f"Model pusher config {model_pusher_config}")
            return model_pusher_config
        except Exception as e:
            raise AdultException(e,sys) from e



    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR,training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR])
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"Training pipleine config: {training_pipeline_config}")
            return training_pipeline_config

        except Exception as e:
            raise AdultException(e,sys) from e

            