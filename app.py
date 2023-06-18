import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    #example = {"Day": int_features[0],"Month":int_features[1], "Year": int_features[2], "Time":int_features[3]}
    #example = pd.DataFrame(example)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    cols = ['Basement','Main_floor', "Second_floor","Third_floor","Fourth_floor"]
    output = [np.round(x) for x in prediction]
    example = pd.DataFrame(data= output,columns=cols)

    return render_template('index.html',date = f"{int_features[1]}/{int_features[0]}/{int_features[2]} - {int_features[3]}pm", b_pred = output[0][0], m_pred = output[0][1], s_pred = output[0][2], t_pred = output[0][3], f_pred = output[0][4])

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    prediction = pd.DataFrame(prediction)
    output = prediction[0]
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)