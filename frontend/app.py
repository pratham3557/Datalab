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
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, r2_score
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier

        st.subheader("AutoML Training")

        target = st.selectbox("Select Target Column", df.columns)

        if st.button("Run AutoML"):

            X = df.drop(columns=[target])
            y = df[target]

            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            results = {}

            if y.nunique() < 20:
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(),
                    "SVM": SVC(),
                    "KNN": KNeighborsClassifier()
                }

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    acc = accuracy_score(y_test, pred)
                    results[name] = acc
    
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor()
                }
    
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    score = r2_score(y_test, pred)
                    results[name] = score
    
            st.subheader("Model Leaderboard")
            st.write(results)


