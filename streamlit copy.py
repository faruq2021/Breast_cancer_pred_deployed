import pandas as pd
import numpy as np
import pickle
import streamlit

model=pickle.load(open('model.pkl', 'rb'))
def bc_prediction(var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8,var_9,var_10):
    pred_arr=np.array([var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8,var_9,var_10]) 
    preds=pred_arr.reshape(1,-1)
    preds=preds.astype(int)
    model_prediction=model.predict(preds)
    return model_prediction

def run():


    streamlit.title("")
    html_temp=""" 
    """
    streamlit.markdown(html_temp)
var_1=streamlit.text_input("Variable 1")
var_2=streamlit.text_input("Variable 2")
var_3=streamlit.text_input("Variable 3")
var_4=streamlit.text_input("Variable 4")
var_5=streamlit.text_input("Variable 5")
var_6=streamlit.text_input("Variable 6")
var_7=streamlit.text_input("Variable 7")
var_8=streamlit.text_input("Variable 8")
var_9=streamlit.text_input("Variable 9")
var_10=streamlit.text_input("Variable 10")

prediction=""
if streamlit.button('Predict'):
        prediction=bc_prediction(var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8,var_9,var_10)

streamlit.success(f'The prediction by Model: {prediction}')

if __name__=='__main__':
    run()


