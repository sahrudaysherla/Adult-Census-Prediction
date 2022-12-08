from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact", 
                                    ["train_data_path","test_data_path","is_ingested","message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
                                    ["schema_file_path","report_file_path","report_page_file_path","is_validated","message"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",
                                        ["transformed_train_data_path", "transformed_test_data_path", "is_transformed", "preprocessed_file_path"])

ModelTrainerArtifact = namedtuple("ModelTrainerArtifact",
                                    ["is_trained", "trained_model_file_path", "train_accuracy", "test_accuracy", "model_accuracy", "test_rmse", "train_rmse", "message"])


ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact", ["is_model_accepted", "evaluated_model_path"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact",
                                ["is_model_pusher", "export_model_file_path"])
