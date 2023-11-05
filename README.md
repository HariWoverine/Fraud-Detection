# Fraud-Detection

## Model deployment in Flast API
1. **app.py**: This is the main file for receiving required information for Fraud detection through GUI or API calls and computing the Fraud detection using XGBoost Classifier and returning it.
2. **index.html**- Template folder: This folder contains an HTML template for user input, based on which the model will make Fraud predictions.
3. **requirements.txt**: This file provides packages to install for your web app to run.
4. **XGBoost_classifier_model.pkl** : Best XGBoost model is saved in pickle file for inference pipeline.
5. **label_encoder.pkl** : The target variable "is_attack" encoder is saved in pickle file for inference pipeline.
## How to Run the model API
1. run the following code in command line.
```
python app.py
```
2. Open the following link in the browser.
```
127.0.0.1:5000
```
3. Provide the Input data for Fraud detection and click on predict the output.
4. The button will direct to the following link
```
127.0.0.1:5000/predict
```
5. Predict page will provide the predicted output at the bottom of the page.

## SNIPPETS OF THE MODEL AS API IS SHOWN BELOW
[bonus1_api.png](https://github.com/HariWoverine/Fraud-Detection/blob/main/bonus1_api.png)
