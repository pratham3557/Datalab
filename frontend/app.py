import streamlit as st
import pandas as pd

st.set_page_config(page_title="DataLab", layout="wide")

st.title("DataLab - Automated Data Analysis & AutoML")

st.write("Upload a dataset to begin analysis")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Shape")
        st.write(df.shape)

        st.subheader("Column Names")
        st.write(df.columns.tolist())

    with col2:
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.write(df.describe())


