#IMPORTING ALL THE LIBRARIES TO HANDLE AND LOAD THE TRAINED MODEL
import streamlit as st
import joblib
import numpy as np
import pandas as pd

#GIVE BACKGROUND AND STYLING FOR THE APP 
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000; /* Black Background */
        color: white;
    }
    .title {
        color: #4e73df;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-top: 50px;
    }
    .input-container {
        background-color: #1a1a1a; /* Dark Grey Container */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(255, 255, 255, 0.1);
        margin-top: 20px;
        margin-bottom: 40px;
        color: white;
    }
    .stButton>button {
        background-color: #4e73df;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #2e5bff;
    }
    .stTextInput>div>input {
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #444;
        background-color: #222;
        color: white;
    }
    .stTextInput>div>input:focus {
        border-color: #4e73df;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stSuccess {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#TITLE OF THE APP
st.markdown('<h1 class="title">🛒 Product Interaction Prediction</h1>', unsafe_allow_html=True)

#CREATING THE MARKDOWN FOR INPUT USER CONTAINER
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# LOAD THE TRAINED MODEL
# LOAD THE TRAINED MODEL
svd = joblib.load(r"D:\1 DS PROJECTS\DS final project 1\save files.pkl")
interaction_matrix = joblib.load(r"D:\1 DS PROJECTS\DS final project 1\INTRACTION SAVE FILE.pkl")
svd_similarity_df = joblib.load(r"D:\1 DS PROJECTS\DS final project 1\SVD SAVE FILE.pkl")

# CREATING THE SELECT BOX TO ENTER THE INPUT FROM USER
customer_id = st.selectbox("Select Customer ID", interaction_matrix.index)
product_id = st.selectbox("Select Product ID", interaction_matrix.columns)

# FUNCTION TO PREDICT INTERACTION SCORE USING THE TRAINED SVD MODEL
def predict_interaction_score(customer_id, product_id):
    customer_idx = interaction_matrix.index.get_loc(customer_id)
    product_idx = interaction_matrix.columns.get_loc(product_id)
    
    predicted_score = np.dot(svd.components_[:, customer_idx], svd.components_[:, product_idx])
    return predicted_score

# DISPLAY THE ENTERED DATA
st.write("Entered Data:")
st.write({
    "Customer ID": customer_id,
    "Product ID": product_id
})

st.markdown('</div>', unsafe_allow_html=True)

#CREATE THE BUTTON TO PREDICT AND DISPLAY THE OUTPUT WHEN THE BUTTON IS CLICKED
if st.button("🔍 Predict Interaction Score"):
    try:
        with st.spinner('Predicting...'):
            prediction = predict_interaction_score(customer_id, product_id)
            st.markdown(f'<div class="stSuccess">✅ Predicted Interaction Score for Customer {customer_id} and Product {product_id} is: {prediction:.2f}</div>', unsafe_allow_html=True)

            if prediction > 0.6:
                st.markdown('<div class="stSuccess">Target this customer with the product! 🎯</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="stError">Do not target this customer with the product. ❌</div>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f'<div class="stError">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
