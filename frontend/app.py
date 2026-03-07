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
    import plotly.express as px

    st.subheader("Data Visualization")

    numeric_columns = df.select_dtypes(include=['int64','float64']).columns

    if len(numeric_columns) > 0:

        column = st.selectbox("Select column for histogram", numeric_columns)

        fig = px.histogram(df, x=column)

        st.plotly_chart(fig)

        st.subheader("Correlation Heatmap")

        corr = df[numeric_columns].corr()

        fig2 = px.imshow(corr, text_auto=True)

        st.plotly_chart(fig2)
        st.write(df.describe())


