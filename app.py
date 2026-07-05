# ==========================================================
# Iris Flower Classification
# Developed using Streamlit & Scikit-Learn
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.main{
padding-top:1rem;
}

.main-title{
font-size:45px;
font-weight:bold;
text-align:center;
color:#2E8B57;
}

.subtitle{
text-align:center;
font-size:20px;
color:gray;
margin-bottom:25px;
}

.metric-card{
background-color:#f8f9fa;
padding:15px;
border-radius:12px;
box-shadow:2px 2px 10px rgba(0,0,0,0.1);
}

.prediction-card{
padding:20px;
border-radius:15px;
background:#E8F5E9;
font-size:22px;
font-weight:bold;
text-align:center;
color:#1B5E20;
}

.footer{
text-align:center;
color:gray;
font-size:14px;
padding-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

with open("models/iris_model.pkl","rb") as f:
    model = pickle.load(f)

with open("models/label_encoder.pkl","rb") as f:
    encoder = pickle.load(f)

# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history=[]

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌸 Iris Flower Classifier")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.write("**Algorithm:** K-Nearest Neighbors")

st.sidebar.write("**Dataset Size:** 150 Flowers")

st.sidebar.write("**Species:** 3")

st.sidebar.write("**Features:** 4")

st.sidebar.write("**Accuracy:** 100%")

st.sidebar.markdown("---")

st.sidebar.subheader("Flower Species")

st.sidebar.success("🌼 Iris Setosa")

st.sidebar.info("🌺 Iris Versicolor")

st.sidebar.warning("🌸 Iris Virginica")

st.sidebar.markdown("---")

st.sidebar.write(
"""
Developed using

- Python
- Streamlit
- Scikit-Learn
- Plotly
"""
)

# ==========================================================
# TITLE
# ==========================================================

st.markdown(
"<div class='main-title'>🌸 Iris Flower Classification</div>",
unsafe_allow_html=True
)

st.markdown(
"<div class='subtitle'>Predict Iris flower species using Machine Learning</div>",
unsafe_allow_html=True
)

st.markdown("---")
# ==========================================================
# INPUT SECTION
# ==========================================================

st.subheader("🌿 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1,
        help="Typical range: 4.3 - 7.9 cm"
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
        help="Typical range: 2.0 - 4.4 cm"
    )

with col2:

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1,
        help="Typical range: 1.0 - 6.9 cm"
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=5.0,
        value=0.2,
        step=0.1,
        help="Typical range: 0.1 - 2.5 cm"
    )

st.markdown("")

predict = st.button(
    "🔍 Predict Species",
    use_container_width=True
)

# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    sample = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    prediction = model.predict(sample)

    probability = model.predict_proba(sample)

    flower = encoder.inverse_transform(prediction)[0]

    confidence = np.max(probability) * 100

    st.markdown("---")

    st.markdown(
        f"""
        <div class="prediction-card">
        🌸 Predicted Species <br><br>
        {flower}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        f"🎯 Prediction Confidence : {confidence:.2f}%"
    )

# ==========================================================
# FLOWER IMAGE
# ==========================================================

    image_path = None

    if flower == "Iris-setosa":
        image_path = "images/setosa.jpg"

    elif flower == "Iris-versicolor":
        image_path = "images/versicolor.jpg"

    elif flower == "Iris-virginica":
        image_path = "images/virginica.jpg"

    if image_path:

        try:

            image = Image.open(image_path)

            st.image(
                image,
                caption=flower,
                width=350
            )

        except:

            st.warning(
                "Flower image not found. "
                "Create an images folder and add "
                "setosa.jpg, versicolor.jpg and virginica.jpg"
            )

# ==========================================================
# PROBABILITY TABLE
# ==========================================================

    probability_df = pd.DataFrame({

        "Species": encoder.classes_,

        "Probability (%)":
            np.round(
                probability[0] * 100,
                2
            )

    })

    st.subheader("Prediction Probabilities")

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )
    # ==========================================================
# INTERACTIVE PROBABILITY CHART
# ==========================================================

    st.subheader("📊 Prediction Probability Chart")

    fig = px.bar(
        probability_df,
        x="Species",
        y="Probability (%)",
        text="Probability (%)",
        title="Model Confidence for Each Species"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Species",
        yaxis_title="Probability (%)",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# SAVE PREDICTION HISTORY
# ==========================================================

    history_row = {
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width,
        "Prediction": flower,
        "Confidence (%)": round(confidence, 2)
    }

    st.session_state.history.append(history_row)

# ==========================================================
# SHOW HISTORY
# ==========================================================

st.markdown("---")

st.subheader("📝 Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No predictions made yet.")

# ==========================================================
# DOWNLOAD HISTORY
# ==========================================================

if len(st.session_state.history) > 0:

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="iris_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

# ==========================================================
# CLEAR HISTORY BUTTON
# ==========================================================

if len(st.session_state.history) > 0:

    if st.button(
        "🗑️ Clear Prediction History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.success("Prediction history cleared successfully!")

        st.rerun()

# ==========================================================
# QUICK FACTS
# ==========================================================

st.markdown("---")

st.subheader("📚 About the Iris Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Samples", "150")

with col2:
    st.metric("Features", "4")

with col3:
    st.metric("Classes", "3")

st.info(
    """
The Iris dataset is one of the most famous datasets in Machine Learning.

It contains measurements of iris flowers belonging to three different
species:

• Iris-setosa

• Iris-versicolor

• Iris-virginica

The model predicts the flower species using four input measurements.
"""
)
# ==========================================================
# INPUT GUIDELINES
# ==========================================================

st.markdown("---")

st.subheader("📌 Input Guidelines")

guide_col1, guide_col2 = st.columns(2)

with guide_col1:

    st.success("""
### Typical Feature Ranges

• Sepal Length : 4.3 – 7.9 cm

• Sepal Width : 2.0 – 4.4 cm

• Petal Length : 1.0 – 6.9 cm

• Petal Width : 0.1 – 2.5 cm
""")

with guide_col2:

    st.info("""
### Flower Species

🌼 Iris-setosa

🌺 Iris-versicolor

🌸 Iris-virginica

Enter values within the above ranges for the most realistic predictions.
""")

# ==========================================================
# MODEL DETAILS
# ==========================================================

st.markdown("---")

st.subheader("🤖 Model Information")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:

    st.metric(
        label="Algorithm",
        value="KNN"
    )

with model_col2:

    st.metric(
        label="Test Accuracy",
        value="100%"
    )

with model_col3:

    st.metric(
        label="Classes",
        value="3"
    )

st.write("")

st.write("""
This application uses the **K-Nearest Neighbors (KNN)** classification
algorithm trained on the famous **Iris Dataset**.

The model predicts the flower species using:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The prediction is based on similarity with previously seen flower
measurements.
""")

# ==========================================================
# TECHNOLOGIES USED
# ==========================================================

st.markdown("---")

st.subheader("🛠️ Technologies Used")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.success("Python")

with tech2:
    st.success("Scikit-Learn")

with tech3:
    st.success("Streamlit")

with tech4:
    st.success("Plotly")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;
padding:20px;
border-radius:12px;
background-color:#f5f5f5;'>

<h3>🌸 Iris Flower Classification</h3>

<p>
Machine Learning Classification Project developed using
<strong>Python</strong>,
<strong>Scikit-Learn</strong>,
<strong>Streamlit</strong>,
and
<strong>Plotly</strong>.
</p>

<p>
Designed for educational and portfolio purposes.
</p>

</div>
""",
unsafe_allow_html=True
)

st.write("")

st.caption(
"© 2026 | Developed by Pratima | B.Tech CSE (AI & ML)"
)