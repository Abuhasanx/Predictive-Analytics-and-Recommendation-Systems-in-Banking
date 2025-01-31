import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.title("CUSTOMER SEGMENTATION")

total_transactions = st.number_input("Total Transactions", min_value=0, step=1)
total_amount = st.number_input("Total Amount", min_value=0.0, step=100.0)
avg_transaction_amount = st.number_input("Average Transaction Amount", min_value=0.0, step=10.0)
num_deposits = st.number_input("Number of Deposits", min_value=0, step=1)
num_withdrawals = st.number_input("Number of Withdrawals", min_value=0, step=1)
withdrawals_amount = st.number_input("Withdrawals Amount", min_value=0.0, step=100.0)
deposits_amount = st.number_input("Deposits Amount", min_value=0.0, step=100.0)

st.write("Entered Data:")
st.write({
    "Total Transactions": total_transactions,
    "Total Amount": total_amount,
    "Average Transaction Amount": avg_transaction_amount,
    "Number of Deposits": num_deposits,
    "Number of Withdrawals": num_withdrawals,
    "Withdrawals Amount": withdrawals_amount,
    "Deposits Amount": deposits_amount
})

# Load the trained model
model1 = joblib.load(r"D:\1 DS PROJECTS\DS final project 1\save file.pkl")
scaler = joblib.load(r"D:\1 DS PROJECTS\DS final project 1\scaler.pkl")

# Check if the loaded object has a predict method
if not hasattr(model1, "predict"):
    st.error("Loaded object is not a valid model. Please check the file.")
else:
    if st.button("Predict"):
        input_data = [[total_transactions, total_amount, avg_transaction_amount, num_deposits, num_withdrawals, withdrawals_amount, deposits_amount]]
        input_data = scaler.transform(input_data)  # Apply the same scaling as training
        cluster = model1.predict(input_data)[0]
        st.write(f"The person belongs to Cluster: {cluster}")