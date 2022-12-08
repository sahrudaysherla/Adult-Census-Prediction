from datetime import datetime
import os

def get_curr_timestamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


ROOT_DIR = os.getcwd()
CURRENT_TIME_STAMP = get_curr_timestamp()
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)
# Data Ingestion related variable

DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY = "tgz_download_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"

TRAINING_PIPELINE_CONFIG_KEY = 'training_pipeline_config'
TRAINING_PIPELINE_NAME_KEY = 'pipeline_name'
TRAINING_PIPELINE_ARTIFACT_DIR = 'artifact_dir'

DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_ARTIFACT_DIR_KEY = "data_validation"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME = "report_page_file_name"

SCHEMA_COLUMNS_KEY = "columns"
SCHEMA_NUMERICAL_COLUMNS = "numerical_columns"
SCHEMA_CATEGORICAL_COLUMNS = "categorical_columns"
SCHEMA_TARGET_COLUMN = "target_column"
SCHEMA_DOMAIN_VALUE_KEY = "domain_value"
SCHEMA_WORKCLASS_DOMAIN_VALUE_KEY = "workclass"
SCHEMA_EDUCATION_DOMAIN_VALUE_KEY = "education"
SCHEMA_MERITAL_STATUS_DOMAIN_VALUE_KEY = "merital-status"
SCHEMA_OCCUPATION_DOMAIN_VALUE_KEY = "occupation"
SCHEMA_RELATIONSHIP_DOMAIN_VALUE_KEY = "relationship"
SCHEMA_RACE_DOMAIN_VALUE_KEY = "race"
SCHEMA_SEX_DOMAIN_VALUE_KEY = "sex"
SCHEMA_COUNTRY_DOMAIN_VALUE_KEY = "country"
SCHEMA_SALARY_DOMAIN_VALUE_KEY = "salary"    


DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_ARTIFACT_DIR_KEY = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSED_OBJ_FILE_NAME_KEY = "preprocessed_object_file_name"

MODEL_TRAINING_CONFIG_KEY = "model_training_config"
MODEL_TRAINING_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINING_TRAINED_MODEL_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINING_BASE_CONF_DIR_KEY = "model_config_dir"
MODEL_TRAINING_BASE_CONF_FILE_NAME_KEY = "model_config_file_name"
MODEL_TRAINING_BASE_ACCURACY = "base_accuracy"
MODEL_TRAINING_ARTIFACT_DIR_KEY = "model_trainer"





MODEL_EVALUATION_ARTIFACT_DIR_KEY = "model_evaluation"
MODEL_EVALUATIUON_FILE_NAME_KEY = "model_evaluation_file_name"
MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"


BEST_MODEL_KEY = "best_model"
MODEL_PATH_KEY = "model_path"
HISTORY_KEY = "history"

MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = "model_export_dir"

EXPERIMENT_DIR_NAME="experiment"
EXPERIMENT_FILE_NAME="experiment.csv"
