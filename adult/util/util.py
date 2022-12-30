import yaml
from adult.exception import AdultException
from adult.constant import *
import os, sys
from adult.logger import logging
import pandas as pd
import numpy as np
import dill

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise AdultException(e,sys) from e

def load_np_data(file_path:str)->np.array:
    try:
        logging.info(f"Loading data into numpy array")
        with open(file_path, 'rb') as file:
            return np.load(file, allow_pickle=True)
    except Exception as e:
        raise AdultException(e,sys) from e



def load_data(file_path:str, schema_file_path:str)-> pd.DataFrame:
    try:
        schema_file = read_yaml_file(schema_file_path)
        schema_cols = schema_file[SCHEMA_COLUMNS_KEY]

        df = pd.read_csv(file_path)
        error_message = ""

        for col in df.columns:
            if col in list(schema_cols):
                df[col].astype(schema_cols[col])

            else:
                error_message = f"{[col]} is not in the schema"

        if len(error_message) > 0:
            raise Exception(error_message)

        return df
    except Exception as e:
        raise AdultException(e,sys) from e

def load_object(file_path:str):
    try:
        with open(file_path, "rb") as file:
            return dill.load(file)
    except Exception as e:
        raise AdultException(e,sys) from e

def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise AdultException(e,sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise AdultException(e, sys) from e

def write_yaml_file(file_path:str, data:dict=None):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(data, file)
    except Exception as e:
        raise AdultException(e, sys) from e

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

