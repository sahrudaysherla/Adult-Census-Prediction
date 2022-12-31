import numpy as np
import pandas as pd
from adult.logger import logging, get_log_dataframe
from adult.exception import AdultException
from adult.pipeline.pipeline import pipeline
from adult.config.configuration import Configuration
from adult.constant import CONFIG_DIR
from adult.util.util import get_current_time_stamp, read_yaml_file, write_yaml_file
import os, sys

from flask import Flask, request, Response
import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


from matplotlib.style import context
import json 
from adult.entity.adultIncomeClassif import adultClassifier, adultData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "adult"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


ADULT_DATA_KEY = "adult_data"
ADULT_INCOME_VALUE_KEY = "adultIncomeValue"

app = Flask(__name__)

@app.route('/artifact', defaults={'req_path': 'adult'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("adult", exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    logging.info(abs_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        log_count = 0
        trained_count = 0
        for files in os.walk(LOG_FOLDER_NAME):
            log_count += len(files)
        for files in os.walk(SAVED_MODELS_DIR_NAME):
            trained_count += len(files)
        return render_template('index.html', dashboard=True, log_count=log_count, trained_count= trained_count)
    except Exception as e:
        return str(e)

@app.route('/slider', methods=['GET', 'POST'])
def slider():
    try:
        return render_template('slider.html')
    except Exception as e:
        return str(e)

@app.route('/slider2', methods=['GET', 'POST'])
def slider2():
    try:
        return render_template('slider2.html')
    except Exception as e:
        return str(e)

@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    Pipeline = pipeline(config=Configuration(config_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        Pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": Pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        ADULT_DATA_KEY: None,
        ADULT_INCOME_VALUE_KEY: None
    }

    if request.method == 'POST':
        age = int(request.form['age'])
        workclass = request.form['workclass']
        fnlwgt = int(request.form['fnlwgt'])
        education = request.form['education']
        education_num = request.form['education_num']
        marital_status = request.form['marital_status']
        occupation = request.form['occupation']
        relationship = request.form['relationship']
        race = request.form['race']
        sex = request.form['sex']
        capital_gain = int(request.form['capital_gain'])
        capital_loss = int(request.form['capital_loss'])
        hours_per_week = int(request.form['hours_per_week'])
        country = request.form['country']

        adult_data = adultData(age = age,
                                workclass = workclass,
                                fnlwgt = fnlwgt,
                                education = education,
                                education_num = education_num,
                                marital_status = marital_status,
                                occupation = occupation,
                                relationship = relationship,
                                race = race,
                                sex = sex,
                                capital_gain = capital_gain,
                                capital_loss = capital_loss,
                                hours_per_week = hours_per_week,
                                country = country
                                   )
        adult_df = adult_data.get_adult_input_data_frame()
        adult_predictor = adultClassifier(model_dir=MODEL_DIR)
        adultIncomeValue = adult_predictor.predict(X=adult_df)
        context = {
            ADULT_DATA_KEY: adult_data.get_adult_data_as_dict(),
            ADULT_INCOME_VALUE_KEY: adultIncomeValue,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)




if __name__ == "__main__":
    app.run(debug=True)




# import numpy as np
# import pandas as pd
# from adult.logger import logging
# from adult.exception import AdultException
# from adult.pipeline.pipeline import pipeline
# from adult.config.configuration import Configuration
# import os, sys

# def main():
#     try:
#         config_path = os.path.join("config","config.yaml")
#         Pipeline = pipeline(Configuration(config_file_path=config_path))
#         Pipeline.run()
#         logging.info("main function execution completed.")
#     except Exception as e:
#             raise AdultException(e,sys) from e

# if __name__ == "__main__":
#     main()


