import pandas as pd
from adult.entity.config_entity import DataIngestionConfig
from adult.entity.artifact_entity import DataIngestionArtifact
from adult.logger import logging
from adult.exception import AdultException
from adult.constant import *
from six.moves import urllib_request
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tarfile
import os,sys

class data_ingestion_component:
    def __init__(self,data_ingestion_config:DataIngestionConfig) -> None:
        try:
            logging.info(f"{'--'*10} Data Ingestion Log Started {'--'*10}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AdultException(e,sys) from e

    def download_data_file(self)->str:
        try:
            download_url = self.data_ingestion_config.dataset_download_url
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            os.makedirs(tgz_download_dir,exist_ok=True)
            adult_file_name = os.path.basename(download_url)
            tgz_file_dir = os.path.join(tgz_download_dir,adult_file_name)
            logging.info(f"{adult_file_name} is getting downloaded from {download_url}")
            urllib_request.urlretrieve(download_url, tgz_file_dir)
            logging.info(f"{adult_file_name} has been successfully downloaded in {tgz_file_dir}")
            return tgz_file_dir
        except Exception as e:
            raise AdultException(e,sys) from e

    def extract_tgz_file(self, tgz_file_path:str):
        try:
            raw_data_path  = self.data_ingestion_config.raw_data_dir
            os.makedirs(raw_data_path, exist_ok=True)
            logging.info(f"Extraction of tgz file is getting executed")
            with tarfile.open(tgz_file_path) as file:
                file.extractall(raw_data_path)
            logging.info(f"Extraction has been completed successfully")
        except Exception as e:
            raise AdultException(e,sys) from e

    def train_test_split(self)->DataIngestionArtifact:
        try:
            raw_data_path = self.data_ingestion_config.raw_data_dir
            csv_file_name = os.listdir(raw_data_path)[0]
            adult_file_dir = os.path.join(raw_data_path,csv_file_name) 
            df = pd.read_csv(adult_file_dir)
            strat_test_set = None
            strat_test_set = None
            # input_features = df[:,:-1]
            # output_features = df[:,-1]
            Split = StratifiedShuffleSplit(n_splits=10,test_size=0.3)
            for train_index,test_index in Split.split(X=df,y=df['salary']):
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,csv_file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, csv_file_name)

            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting Training Dataset to {train_file_path}")
                strat_train_set.to_csv(train_file_path,index=False)
            
            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir,exist_ok=True)
                logging.info(f"Exporting Testing Dataset to {test_file_path}")
                strat_test_set.to_csv(test_file_path,index=False)
            data_ingestion_artifact = DataIngestionArtifact(train_data_path = train_file_path,
                                                            test_data_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data Ingestion Completed Successfully")
            return data_ingestion_artifact
        except Exception as e:
            raise AdultException(e,sys) from e

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            download = self.download_data_file()
            self.extract_tgz_file(tgz_file_path=download)
            return self.train_test_split()
        except Exception as e:
            raise AdultException(e,sys) from e
            