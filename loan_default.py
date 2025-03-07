import streamlit as st
import pandas as pd
import joblib 

st.title("LOAN DEFAULT")


age = st.number_input("Age", min_value=18, max_value=100, step=1)
income = st.number_input("Income", min_value=0.0, step=100.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
loan_term = st.number_input("Loan Term (in months)", min_value=1, step=1)
loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)
monthly_installment = st.number_input("Monthly Installment", min_value=0.0, step=10.0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)


st.write("Entered Data:")
st.write({
    "Age": age,
    "Income": income,
    "Credit Score": credit_score,
    "Loan Term": loan_term,
    "Loan Amount": loan_amount,
    "Monthly Installment": monthly_installment,
    "Interest Rate": interest_rate

})


# Load the trained model
model = joblib.load(r"D:\1 DS PROJECTS\DS final project 1.pkl")

if st.button("Predict"):
    input_data = [[age, income, credit_score, loan_term, loan_amount, monthly_installment, interest_rate]]
    prediction = model.predict(input_data)[0]
    if prediction == 0:
        st.error("The person is likely to default on the loan.")
    else:
        st.success("The person is likely to repay the loan.")
df = pd.read_csv(r"D:\1 DS PROJECTS\DS final project 1\loan_default_data.csv")
st.dataframe(df)



   