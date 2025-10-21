import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib
import altair as alt

st.set_page_config(page_title="Accident Risk Predictor", layout="wide", page_icon="🚦")

st.title("🚦 Accident Risk Prediction App")
st.markdown("Predict accident risk based on road, weather, and traffic conditions.")

# --- Load train.csv ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('train.csv')
        return df
    except Exception as e:
        st.error(f"Could not load train.csv: {e}")
        return None

train_df = load_data()

if train_df is not None:
    st.subheader("📊 Dataset Insights")

    # Total records and features
    st.markdown(f"**Total records:** {train_df.shape[0]}")
    st.markdown(f"**Total features:** {train_df.shape[1]}")

    # 1️⃣ Accident Risk Distribution
    st.markdown("**Accident Risk Distribution:**")
    chart1 = alt.Chart(train_df).mark_bar(color="#F28E2B").encode(
        x=alt.X("accident_risk", bin=alt.Bin(maxbins=20), title="Accident Risk"),
        y=alt.Y("count()", title="Count"),
        tooltip=["count()"]
    ).properties(height=250)
    st.altair_chart(chart1, use_container_width=True)

    # 2️⃣ Average Accident Risk per Lighting Condition
    st.markdown("**Average Accident Risk by Lighting Condition:**")
    lighting_avg = train_df.groupby("lighting")["accident_risk"].mean().reset_index()
    chart2 = alt.Chart(lighting_avg).mark_bar(color="#4E79A7").encode(
        x=alt.X("lighting:N", title="Lighting", sort='-y', axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accident_risk:Q", title="Average Accident Risk"),
        tooltip=["lighting", "accident_risk"]
    ).properties(height=250)
    st.altair_chart(chart2, use_container_width=True)


    # --- Preprocessing ---
    features = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
                'weather', 'road_signs_present', 'public_road', 'time_of_day',
                'holiday', 'school_season', 'num_reported_accidents']
    target = 'accident_risk'

    X = train_df[features]
    y = train_df[target]

    # Encode categorical features
    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # --- Train XGB Model ---
    @st.cache_resource
    def train_model(X, y):
        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',
            tree_method='hist'
        )
        model.fit(X, y)
        return model

    model = train_model(X, y)

    # --- Sidebar Inputs ---
    st.sidebar.header("🧭 Set Input Features")
    user_input = {}

    for col in features:
        if col in encoders:  # categorical
            vals = train_df[col].astype(str).unique()
            selected = st.sidebar.selectbox(col, vals)
            user_input[col] = encoders[col].transform([selected])[0]
        else:  # numeric
            min_val = int(train_df[col].min())
            max_val = int(train_df[col].max())
            val = st.sidebar.slider(col, min_val, max_val, int(train_df[col].median()))
            user_input[col] = val

    input_df = pd.DataFrame([user_input])


    # --- Predict ---
    if st.button("🚀 Predict Accident Risk"):
        prediction = model.predict(input_df)[0]
        st.success(f"### ✅ Predicted Accident Risk: {prediction:.3f}")
        st.progress(float(np.clip(prediction, 0.0, 1.0)))  # safe for Streamlit


    # --- Feature Importance ---
    st.markdown("---")
    st.subheader("🌟 Feature Importance")
    importance_df = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    chart = alt.Chart(importance_df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x='importance',
        y=alt.Y('feature', sort='-x'),
        color='importance',
        tooltip=['feature', 'importance']
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

# st.caption("Developed by Simron ❤️ using Streamlit | 2025 Kaggle Playground Series")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 14px;'>"
    "Developed by Simron ❤️ using Streamlit | 2025 Kaggle Playground Series"
    "</p>",
    unsafe_allow_html=True
)
