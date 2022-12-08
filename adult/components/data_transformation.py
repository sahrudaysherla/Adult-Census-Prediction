from adult.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact
from adult.entity.config_entity import DataTransformationConfig
import numpy as np
import pandas as pd
from adult.constant import *
from adult.util.util import read_yaml_file, load_data, save_numpy_array_data, save_object
from adult.logger import logging
from adult.exception import AdultException
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import os,sys

class data_transformation_component:
    def __init__(self, data_ingestion_artifact = DataIngestionArtifact, data_validation_artifact = DataValidationArtifact, data_transformation_config = DataTransformationConfig) -> None:
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise AdultException(e,sys) from e

    def get_transformed_object(self)->ColumnTransformer:
        try: 
            schema_data = self.data_validation_artifact.schema_file_path
            data_schema = read_yaml_file(schema_data)

            num_cols = data_schema[SCHEMA_NUMERICAL_COLUMNS]
            cat_cols = data_schema[SCHEMA_CATEGORICAL_COLUMNS]
            

            num_pipeline = make_pipeline(SimpleImputer(missing_values=np.nan, strategy="mean"),StandardScaler() )
            cat_pipeline = make_pipeline(SimpleImputer(missing_values=np.nan, strategy="most_frequent"), OneHotEncoder(sparse=False,handle_unknown='ignore'), StandardScaler(with_mean=False))
            

            logging.info(f"Numerical Columns are {num_cols}")
            logging.info(f"Categorical Columns are {cat_cols}")
            preprocessing = ColumnTransformer([("num_pipeline", num_pipeline, num_cols),
                                                ("cat_cols", cat_pipeline, cat_cols),])
            return preprocessing

        except Exception as e:
            raise AdultException(e,sys) from e        
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object")

            preprocessing_obj = self.get_transformed_object()
            label = LabelEncoder()

            logging.info(f"Obtaining training and test file path.")

            train_file_path = self.data_ingestion_artifact.train_data_path
            test_file_path = self.data_ingestion_artifact.test_data_path
            schema_file_path = self.data_validation_artifact.schema_file_path

            #print(train_file_path, test_file_path, schema_file_path)

            logging.info(f"Loading training and test data as pandas dataframe.")

            train_df = load_data(file_path = train_file_path, schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path,schema_file_path=schema_file_path)
            schema = read_yaml_file(schema_file_path)

            target_column = schema[SCHEMA_TARGET_COLUMN]

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            
            
            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            
            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            target_feature_train_arr = label.fit_transform(target_feature_train_df)
            target_feature_test_arr = label.transform(target_feature_test_df)
            train_arr = np.c_[ input_feature_train_arr, target_feature_train_arr]

            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            transformed_train_dir = self.data_transformation_config.transformed_train_file_path
            transformed_test_dir = self.data_transformation_config.transformed_test_file_path
            
            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")
            
            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)
            
            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)
            
            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(transformed_train_data_path=transformed_train_file_path,
                                                                    transformed_test_data_path=transformed_test_file_path,
                                                                    preprocessed_file_path=preprocessing_obj_file_path,
                                                                    is_transformed=True)
            return data_transformation_artifact

            

        except Exception as e:
            raise AdultException(e,sys) from e





