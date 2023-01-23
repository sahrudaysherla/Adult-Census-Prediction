# Adult Census Income Prediction
## This is a Machine Learning Project to classify a person's Income based on the certain input parameters, it is designed to have GUI.
## Anybody can train this model on their personal preferance to the parameters in the Update Parameters section

## This model is trained on the dataset:
[Click here for dataset](https://www.kaggle.com/datasets/uciml/adult-census-income)

### Run the following code in the terminal to install the required files and libraries
```
python setup.py install
```
```
pip install -r requirements.txt
```
### Tests for Data drift is also performed using [Evidently](https://github.com/evidentlyai/evidently) for each training Iteration. Report for the data drift can be found at the following path of the project:

```
/workspaces/Adult-Census-Prediction/adult/artifact/data_validation/{latest timestamp of the training Iteration}
```
> :warning: Report file for Data Drift will only be generated when you train the model

### Run the following code in terminal to start the server
``` 
python app.py 
```
### once you start the server the below dashboard will be shown on the server port.

### Following is the dashboard UI, feel free to walk through each and every url.
#### Model will start training with just the click of the **Let's Train** button
![Dashboard](https://github.com/sahrudaysherla/Adult-Census-Prediction/raw/main/showcase/dashboard.png)

### Here Artifacts of the model can be found for each Iteration or training
![Artifacts](https://github.com/sahrudaysherla/Adult-Census-Prediction/raw/main/showcase/artifact.png)

### Following is the Classifier UI one can pass the inputs and the model will classify their Income
![Classifier image](https://github.com/sahrudaysherla/Adult-Census-Prediction/raw/main/showcase/classifier.png)

![Classifier 2](https://github.com/sahrudaysherla/Adult-Census-Prediction/raw/main/showcase/classifier2.png)

### Some insights about the dataset can be found here in this section
![Insights](https://github.com/sahrudaysherla/Adult-Census-Prediction/raw/main/showcase/insights.png)

### Performance of different algorithms tested on the dataset can be found here
![Matrix](https://github.com/sahrudaysherla/Adult-Census-Prediction/raw/main/showcase/matrix.png)

### Updates about the model in training mode can be found in this section
![Train](https://github.com/sahrudaysherla/Adult-Census-Prediction/raw/main/showcase/train.png)

### Following are the trained models with the best accuracy
![Trained Models](https://github.com/sahrudaysherla/Adult-Census-Prediction/raw/main/showcase/trainedModels.png)

### One can tweak and update parameters and train the same model on them for the optimum results
![Update Parameters](https://github.com/sahrudaysherla/Adult-Census-Prediction/raw/main/showcase/updateParameters.png)

