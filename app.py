import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
filename = 'rf_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a Streamlit app
st.title("Credit Card Default Prediction")

# Create input fields for user data
st.header("Enter Customer Information:")

limit_bal = st.number_input("Credit Limit:", min_value=0)
sex = st.selectbox("Gender:", [1, 2])
education = st.selectbox("Education Level:", [1, 2, 3, 4])
marriage = st.selectbox("Marital Status:", [1, 2, 3])
age = st.number_input("Age:", min_value=18)
pay_1 = st.selectbox("Repayment Status (Sep):", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
pay_2 = st.selectbox("Repayment Status (Aug):", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
pay_3 = st.selectbox("Repayment Status (Jul):", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
pay_4 = st.selectbox("Repayment Status (Jun):", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
pay_5 = st.selectbox("Repayment Status (May):", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
pay_6 = st.selectbox("Repayment Status (Apr):", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
bill_amt1 = st.number_input("Bill Amount (Sep):", min_value=0)
bill_amt2 = st.number_input("Bill Amount (Aug):", min_value=0)
bill_amt3 = st.number_input("Bill Amount (Jul):", min_value=0)
bill_amt4 = st.number_input("Bill Amount (Jun):", min_value=0)
bill_amt5 = st.number_input("Bill Amount (May):", min_value=0)
bill_amt6 = st.number_input("Bill Amount (Apr):", min_value=0)
pay_amt1 = st.number_input("Previous Payment (Sep):", min_value=0)
pay_amt2 = st.number_input("Previous Payment (Aug):", min_value=0)
pay_amt3 = st.number_input("Previous Payment (Jul):", min_value=0)
pay_amt4 = st.number_input("Previous Payment (Jun):", min_value=0)
pay_amt5 = st.number_input("Previous Payment (May):", min_value=0)
pay_amt6 = st.number_input("Previous Payment (Apr):", min_value=0)

# Create a button to make predictions
if st.button("Predict"):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'LIMIT_BAL': [limit_bal],
        'SEX': [sex],
        'EDUCATION': [education],
        'MARRIAGE': [marriage],
        'AGE': [age],
        'PAY_0': [pay_1],
        'PAY_2': [pay_2],
        'PAY_3': [pay_3],
        'PAY_4': [pay_4],
        'PAY_5': [pay_5],
        'PAY_6': [pay_6],
        'BILL_AMT1': [bill_amt1],
        'BILL_AMT2': [bill_amt2],
        'BILL_AMT3': [bill_amt3],
        'BILL_AMT4': [bill_amt4],
        'BILL_AMT5': [bill_amt5],
        'BILL_AMT6': [bill_amt6],
        'PAY_AMT1': [pay_amt1],
        'PAY_AMT2': [pay_amt2],
        'PAY_AMT3': [pay_amt3],
        'PAY_AMT4': [pay_amt4],
        'PAY_AMT5': [pay_amt5],
        'PAY_AMT6': [pay_amt6],
    })


    # Make predictions using the loaded model
    prediction = loaded_model.predict(input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.error("Prediction: The customer is likely to default on their credit card.")
    else:
        st.success("Prediction: The customer is likely to pay their credit card on time.")

from google.colab import files
files.download('app.py')
