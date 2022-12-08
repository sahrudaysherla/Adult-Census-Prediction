from collections import namedtuple
DataIngestionConfig=namedtuple("DataIngestionConfig",
["dataset_download_url","tgz_download_dir","raw_data_dir","ingested_train_dir","ingested_test_dir"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",["artifact_dir"])

DataValidationConfig = namedtuple("DataValidationConfig",
                                ['schema_file_path', 'report_file_path', 'report_page_file_path'])

DataTransformationConfig = namedtuple("DataTransformationConfig",
                                    ["transformed_train_file_path", "transformed_test_file_path","preprocessed_file_path"])

ModelTrainerConfig = namedtuple("ModelTrainerConfig",
                                ["base_accuracy", "trained_model_path", "model_config_path"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig",
                                    ["model_evaluation_file_path", "timestamp"])

ModelPusherConfig = namedtuple("ModelPusherConfig",
                                ["export_dir_path"])