import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import warnings
warnings.simplefilter(action="ignore")

app = Flask(__name__)
model = pickle.load(open('XGBoost_classifier_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl','rb'))
output_dictionary = {'True': 'Fraud', 'False': 'Non-Fraud'}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = pd.DataFrame(request.form.values()).T
    data.columns = ['anomaly_feat_0', 'anomaly_feat_1', 'anomaly_feat_2', 'anomaly_feat_3',
       'count_feat_0', 'count_feat_1', 'count_feat_2', 'count_feat_3',
       'count_feat_4', 'count_feat_5', 'count_feat_6', 'count_feat_7',
       'count_feat_8', 'count_feat_9', 'count_feat_10', 'count_feat_11',
       'count_feat_12', 'count_feat_13', 'count_feat_14', 'count_feat_15',
       'count_feat_16', 'count_feat_17', 'interaction_feat_0','interaction_feat_1', 
        'interaction_feat_2', 'interaction_feat_3','timestamp', 'accountid', 
        'device_feat_1', 'device_feat_2']
    data['anomaly_feat_0'] = data['anomaly_feat_0'].astype('bool')
    data['anomaly_feat_1'] = data['anomaly_feat_1'].astype('bool')
    data['anomaly_feat_2'] = data['anomaly_feat_2'].astype('bool')
    data['anomaly_feat_3'] = data['anomaly_feat_3'].astype('bool')
    data['interaction_feat_0'] = data['interaction_feat_0'].astype('bool')
    data['interaction_feat_1'] = data['interaction_feat_2'].astype('bool')
    prediction = model.predict(data)
    output = label_encoder.inverse_transform(prediction)
    return render_template('index.html', prediction_text='Predicted output is {}'.format(output_dictionary[str(output[0])]))
    
if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=5000)