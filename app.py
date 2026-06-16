import streamlit as st
import joblib
import numpy as np
import time
st.set_page_config(
    page_title="🧠 AI Teen Depression Predictor",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    text-align: center;
    color: #2E86C1;
}

h3 {
    color: #117864;
}

.stButton>button {
    background-color: #2E86C1;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

[data-testid="stSidebar"] {
    background-color: #EBF5FB;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This application predicts the likelihood of depression
using a trained Logistic Regression model.

⚠️ This is for educational purposes only and should not
replace professional medical advice.
""")

st.markdown("<h1>🧠 AI Teen Depression Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align:center;color:gray;'>Enter the details below and click Predict</h4>",
    unsafe_allow_html=True
)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")

    age = st.number_input("🎂 Age", 10, 30, 18)

    gender = st.selectbox(
        "👫 Gender",
        ["Male", "Female"]
    )

    daily_social_media_hours = st.slider(
        "📱 Daily Social Media Hours",
        0.0, 15.0, 2.0
    )

    sleep_hours = st.slider(
        "😴 Sleep Hours",
        0.0, 12.0, 7.0
    )

    screen_time_before_sleep = st.slider(
        "💻 Screen Time Before Sleep",
        0.0, 10.0, 1.0
    )

with col2:

    st.subheader("🩺 Health & Lifestyle")

    academic_performance = st.slider(
        "📚 Academic Performance",
        0, 10, 5
    )

    physical_activity = st.slider(
        "🏃 Physical Activity",
        0, 10, 5
    )

    social_interaction_level = st.slider(
        "🗣️ Social Interaction Level",
        0, 10, 5
    )

    stress_level = st.slider(
        "😟 Stress Level",
        0, 10, 5
    )

    anxiety_level = st.slider(
        "😰 Anxiety Level",
        0, 10, 5
    )

    addiction_level = st.slider(
        "🎮 Addiction Level",
        0, 10, 5
    )

platform = st.selectbox(
    "🌐 Platform Usage",
    ["Other", "Both", "Instagram", "TikTok"]
)


gender = 0 if gender == "Male" else 1

platform_usage_Both = 1 if platform == "Both" else 0
platform_usage_Instagram = 1 if platform == "Instagram" else 0
platform_usage_TikTok = 1 if platform == "TikTok" else 0

st.markdown("---")

if st.button("🚀 Predict Depression Risk"):

    input_data = np.array([[
        age,
        gender,
        daily_social_media_hours,
        sleep_hours,
        screen_time_before_sleep,
        academic_performance,
        physical_activity,
        social_interaction_level,
        stress_level,
        anxiety_level,
        addiction_level,
        platform_usage_Both,
        platform_usage_Instagram,
        platform_usage_TikTok
    ]])

    input_scaled = scaler.transform(input_data)

    with st.spinner("🔍 Analyzing data..."):
        time.sleep(2)

    prediction = model.predict(input_scaled)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("🚨 High Risk of Depression Detected")
        st.warning(
            "Please consult a qualified mental health professional for a proper assessment."
        )
    else:
        st.success("✅ Low Risk of Depression Detected")
        st.balloons()


st.markdown("---")
st.markdown(
    """
    <center>
    ❤️ <b>Developed with Streamlit & Scikit-learn</b><br>
    🧠 Teen Depression Prediction Project
    </center>
    """,
    unsafe_allow_html=True
)