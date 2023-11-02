import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Medical Diagonistic app')
st.subheader('Does the patient have diabetes')
df=pd.read_csv('diabetes.csv')
if st.sidebar.checkbox('View Data',False):
    st.write(df)
if st.sidebar.checkbox('View Distributions',False):
    df.hist()
    plt.tight_layout()
    st.pyplot()

    

# step1 : load the pickled model

model = open('rfc.pickle','rb')
clf = pickle.load(model)
model.close()

# step2 : Get the front end users input
pregs = st.number_input('Pregnancies',0,20,0)
plas=st.slider('Glucose',40,200,40)
pres=st.slider('BloodPressure',20,150,20)
skin=st.slider('Skin Thickness',7,99,7)
insulin = st.slider('Insulin',14,850,14)
bmi = st.slider('BMI',18,70,18)
dpf=st.slider ('DiabetesPedigreeFunction', 0.05, 2.50, 0.05)
age=st.slider('Age', 21, 90, 21)

# step3 : get the model input
input_data = [[pregs,plas,pres,skin,insulin,bmi,dpf,age]]

# step4 : get the prediction and print the result
prediction = clf.predict(input_data)[0]
if st.button('Predict'):
    if prediction==0:
        st.subheader('Non Diabetic')
    else:
        st.subheader('Diabetic')
        
