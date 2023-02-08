import pickle
import numpy as np
import streamlit as st
loaded_model = pickle.load(open('model.pkl', 'rb'))
def dia_pred(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
def main():
    st.title('Diabetes Prediction Web App')
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Classifier ML App </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    pregnancies = st.text_input("pregnancies")
    glucose = st.text_input("glucose")
    blood_pressure = st.text_input("blood_pressure")
    skin_thickness = st.text_input("skin_thickness")
    insulin = st.text_input("insulin")
    bmi = st.text_input("bmi")
    diabetes_pedegree_function = st.text_input("dpf")
    age = st.text_input("age")
    result= ''
    if st.button("Predict"):
        result = dia_pred([float(pregnancies), float(glucose), float(blood_pressure), float(skin_thickness), float(insulin), float(bmi), float(diabetes_pedegree_function), float(age)])
    st.success('The output is {}'.format(result))
if __name__=='__main__':
    main()