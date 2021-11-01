import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

"""
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
"""

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if prediction==0:
        print("Malignant")
    else:
        print("Benign")

    

    return render_template('homepage.html', prediction_text='Your prediction is {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
