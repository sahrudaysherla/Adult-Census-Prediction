from adult.logger import logging
from adult.exception import AdultException
from adult.entity.config_entity import DataValidationConfig
from adult.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact
import os,sys
from adult.util.util import read_yaml_file
import pandas  as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from adult.constant import *
import json
class data_validation_component:
    def __init__(self, data_ingestion_artifact =DataIngestionArtifact , data_validation_config = DataValidationConfig) -> None:
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise AdultException(e,sys) from e
        
    def get_train_and_test_data(self):
        try:
            train_data = pd.read_csv(self.data_ingestion_artifact.train_data_path)
            test_data  = pd.read_csv(self.data_ingestion_artifact.test_data_path)
            return train_data, test_data
        except Exception as e:
            raise AdultException(e,sys) from e

    def is_train_and_test_file_exist(self)->bool:
        try:
            train_data_test = False
            test_data_test = False
            train_data_test = os.path.exists(self.data_ingestion_artifact.train_data_path)
            test_data_test = os.path.exists(self.data_ingestion_artifact.test_data_path)
            
            available = train_data_test and test_data_test
            
            if not available:
                training_file_path = self.data_ingestion_artifact.train_data_path
                testing_file_path = self.data_ingestion_artifact.test_file_path
                message = f"Train file at {training_file_path} or Test file at {testing_file_path} is missing"
                logging.info(message)
            return available
        except Exception as e:
            raise AdultException(e,sys) from e
    
    def validate_data_schema(self)->bool:
        try:
            self.schema_file_path = self.data_validation_config.schema_file_path
            validation_status = False
            column_count_equal = False
            column_names_same = False
            schema_data = read_yaml_file(self.schema_file_path)
            schema_columns = schema_data[SCHEMA_COLUMNS_KEY]
            train_data = pd.read_csv(self.data_ingestion_artifact.train_data_path)

            if len(list(schema_columns.keys())) == len(train_data.columns):
                column_count_equal = True
            
            for column in train_data.columns:
                    if column in list(schema_columns.keys()):
                        train_data[column].astype(schema_columns[column])
                        column_names_same = True
                    else:
                        column_names_same = False
            
            validation_status = column_count_equal and column_names_same

            return validation_status
        except Exception as e:
            raise AdultException(e,sys) from e

    def get_and_save_data_drift_report(self):
        try:
            ca_data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            train_data,test_data = self.get_train_and_test_data()
            ca_data_drift_profile.calculate(train_data, test_data)
            report = json.loads(ca_data_drift_profile.json())

            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir,exist_ok=True)

            with open(report_file_path,"w") as report_file:
                json.dump(report, report_file, indent=6)
            return report
        except Exception as e:
            raise AdultException(e,sys) from e
    
    def save_data_drift_report_page(self):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df,test_df = self.get_train_and_test_data()
            dashboard.calculate(train_df,test_df)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir,exist_ok=True)
            dashboard.save(report_page_file_path)
        except Exception as e:
            raise AdultException(e,sys) from e

            
    def is_data_drift_found(self)->bool:
        try:

            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise AdultException(e,sys) from e
    def initiate_data_validation(self)->DataValidationArtifact :
        try:
            self.is_train_and_test_file_exist()
            self.is_data_drift_found()
            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=self.validate_data_schema(),
                message="Data Validation performed successully."
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise AdultException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Valdaition log completed.{'<<'*30} \n\n")






