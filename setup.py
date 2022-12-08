from setuptools import setup,find_packages
from typing import List

PROJECT_NAME = "AdultCensusIncomePrediction"
VERSION = "0.0.1"
AUTHOR = "Sahruday Sherla"
DESCRIPTION = "This project classifies the if the adult income is less or greater than 50K"
REQUIREMENT_FILE_NAME = "requirements.txt"
HYPEN_E_DOT = "-e ."

def get_requirements() ->List[str]:
    with open(REQUIREMENT_FILE_NAME) as req_file:
        req_file = req_file.readlines()
        req_file = [lines.replace("\n","") for lines in req_file]
        if HYPEN_E_DOT in req_file:
            req_file.remove(HYPEN_E_DOT)
    
        return req_file

setup(
    name = PROJECT_NAME,
    version= VERSION,
    description = DESCRIPTION,
    author=AUTHOR,
    packages=find_packages(),
    install_requires = get_requirements())