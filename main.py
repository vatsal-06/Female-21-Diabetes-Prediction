import streamlit as st
import pickle

pickle_in = open('logisticRegr.pkl', 'rb')
classifier = pickle.load(pickle_in)

st.title('Diabetes Prediction')

st.title('Diabetes Prediction (Only for females above 21 Years of Age)')

name = st.text_input('Name: ')

pregnancy = st.number_input('No. of Pregnancies: ')
glucose = st.number_input('Glucose Level: ')
bp = st.number_input('Diastolic Blood Pressure (mm/Hg): ')
skin = st.number_input('Triceps Skin Fold Thickness (mm): ')
insulin = st.number_input('Two-Hour Serum Insulin (mu U/ml):')
bmi = st.number_input('Body Mass Index (Weight in kg/(Height in m)^2):')
dpf = st.number_input('Diabetes Pedigree Function: ')
age = st.number_input('Age: ')

submit = st.button('Submit')

if submit:
    prediction = classifier.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])

    if prediction == 0:
        st.write('Congratulations!', name, 'you are not diabetic')
    else:
        st.write(name, " we are really sorry to say but it seems like you are Diabetic.")
