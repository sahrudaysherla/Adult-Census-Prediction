import numpy as np
import pandas as pd
from adult.logger import logging
from adult.exception import AdultException
from adult.pipeline.pipeline import pipeline
from adult.config.configuration import Configuration
import os, sys

def main():
    try:
        config_path = os.path.join("config","config.yaml")
        Pipeline = pipeline(Configuration(config_file_path=config_path))
        Pipeline.run()
        logging.info("main function execution completed.")
    except Exception as e:
            raise AdultException(e,sys) from e

if __name__ == "__main__":
    main()