from fastapi import FastAPI
from starlette.routing import Host
import uvicorn
from feature_matrix import features
import pickle
import numpy as np
import pandas as pd
#create an app object
app=FastAPI()
pickle_in=open('model.pkl','rb')
bc_predictor=pickle.load(pickle_in)

#create index route that opens automatically on the ip address
@app.get('/')
def index():
    return{'message':'Hello'}
@app.get('/welcome')
def get_name(name:'str'):
    return{'Welcome to breast cancer predicttion':f'{name}'}
@app.post('/predict')
def predict_bc(data:features):
    data=data.dict()
    radius_mean=data['radius_mean']
    
    texture_mean=data['texture_mean']
    
    perimeter_mean=data['perimeter_mean']
    
    area_mean=data['area_mean']
    
    smoothness_mean=data['smoothness_mean']
    
    compactness_mean=data['compactness_mean']
    
    concavity_mean=data['concavity_mean']
    
    concave_points_mean=data['concave_points_mean']
   
    symmetry_mean=data['symmetry_mean']
    
    fractal_dimension_mean=data['fractal_dimension_mean']
    
    #print(bc_predictor.predict(([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean]]))
   
    prediction=bc_predictor.predict(([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean]]))
    if(prediction==0):
        prediction='Malignant'
    else:
        prediction='Benign'
    return{'prediction':prediction}



if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1.port=8000')

