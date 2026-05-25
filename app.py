import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------------
# Load Trained Model
# -----------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "titanic_ann_model.h5",
        compile=False
    )
    return model

model = load_model()

# -----------------------------------
# Header Section
# -----------------------------------
st.markdown("""
# 🚢 Titanic Survival Prediction System
### Deep Learning Based Passenger Survival Prediction
""")

col1, col2 = st.columns([3, 1])

with col1:
    st.success("""
This AI system predicts whether a Titanic passenger
would survive or not using an Artificial Neural Network (ANN)
trained using TensorFlow and deployed with Streamlit.
""")

with col2:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/854/854878.png",
        width=140
    )

st.divider()

# -----------------------------------
# Project Description
# -----------------------------------
st.subheader("📌 Project Description")

st.write("""
This application predicts the survival chances of a Titanic passenger.

### Purpose of Application
- Predict passenger survival
- Use passenger information for inference
- Demonstrate Deep Learning deployment

### ANN Usage
The model uses an Artificial Neural Network (ANN) with:
- Input Layer
- Hidden Layer
- Output Layer

### TensorFlow Deployment
The trained TensorFlow/Keras model is loaded
and used for real-time prediction.
""")

st.divider()

# -----------------------------------
# Passenger Input Form
# -----------------------------------
st.subheader("🧾 Passenger Input Form")

container = st.container(border=True)

with container:

    col1, col2, col3 = st.columns(3)

    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3]
        )

    with col2:
        age = st.slider(
            "Age",
            min_value=1,
            max_value=80,
            value=24
        )

    with col3:
        fare = st.number_input(
            "Fare",
            min_value=0.0,
            max_value=600.0,
            value=120.0,
            step=1.0
        )

st.divider()

# -----------------------------------
# Prediction Button
# -----------------------------------
if st.button("🔍 Predict Survival", use_container_width=True):

    # -----------------------------------
    # Data Preprocessing
    # Same preprocessing used in training
    # -----------------------------------

    pclass_norm = (pclass - 1) / (3 - 1)
    age_norm = age / 80
    fare_norm = fare / 600

    input_data = np.array([
        [pclass_norm, age_norm, fare_norm]
    ])

    # -----------------------------------
    # Prediction
    # -----------------------------------
    prediction = model.predict(input_data)

    probability = prediction[0][0]

    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    confidence_score = probability * 100

    st.divider()

    # -----------------------------------
    # Prediction Output Area
    # -----------------------------------
    st.subheader("📊 Prediction Output")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Prediction Result",
            value=result
        )

    with col2:
        st.metric(
            label="Survival Probability",
            value=f"{probability:.2%}"
        )

    with col3:
        st.metric(
            label="Confidence Score",
            value=f"{confidence_score:.2f}%"
        )

    st.divider()

    # -----------------------------------
    # Visualization Area
    # -----------------------------------
    st.subheader("📈 Prediction Visualization")

    survive_prob = probability
    nonsurvive_prob = 1 - probability

    chart_data = pd.DataFrame({
        "Category": ["Survive", "Not Survive"],
        "Probability": [
            survive_prob,
            nonsurvive_prob
        ]
    })

    col1, col2 = st.columns(2)

    # Bar Chart
    with col1:
        st.write("### Bar Chart")

        fig, ax = plt.subplots()

        ax.bar(
            chart_data["Category"],
            chart_data["Probability"]
        )

        ax.set_ylabel("Probability")
        ax.set_title("Survival Probability")

        st.pyplot(fig)

    # Pie Chart
    with col2:
        st.write("### Pie Chart")

        fig2, ax2 = plt.subplots()

        ax2.pie(
            [survive_prob, nonsurvive_prob],
            labels=["Survive", "Not Survive"],
            autopct="%1.1f%%"
        )

        ax2.set_title("Prediction Distribution")

        st.pyplot(fig2)

    # Probability Meter
    st.subheader("🎯 Survival Probability Meter")
    st.progress(float(probability))

    st.info(
        f"Predicted survival probability: "
        f"{probability:.2%}"
    )
