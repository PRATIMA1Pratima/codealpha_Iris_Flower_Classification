# 🌸 Iris Flower Classification using Machine Learning

## 📌 Project Overview

The **Iris Flower Classification** project is a Machine Learning classification application that predicts the species of an Iris flower based on its physical measurements.

The model is trained using the famous **Iris Dataset** and classifies flowers into one of the following three species:

* 🌼 Iris-setosa
* 🌺 Iris-versicolor
* 🌸 Iris-virginica

The project includes a professional **Streamlit web application** that allows users to enter flower measurements and receive instant predictions with confidence scores and visualizations.

---

# 🚀 Features

* Professional Streamlit Web Application
* Predicts Iris flower species
* K-Nearest Neighbors (KNN) Classification Model
* Interactive and user-friendly interface
* Prediction confidence score
* Probability distribution chart
* Prediction history
* Download prediction history as CSV
* Responsive layout
* Flower image display
* Professional project structure
* Ready for GitHub and Streamlit Cloud deployment

---

# 📂 Project Structure

```text
Iris_Flower_Classification/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── Iris.csv
│
├── images/
│   ├── setosa.jpg
│   ├── versicolor.jpg
│   └── virginica.jpg
│
├── models/
│   ├── iris_model.pkl
│   └── label_encoder.pkl
│
└── notebook/
    └── Iris_Flower_Classification.ipynb
```

---

# 📊 Dataset

The project uses the famous **Iris Dataset**, which contains measurements of Iris flowers.

### Features

* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

### Target Variable

Species

* Iris-setosa
* Iris-versicolor
* Iris-virginica

### Dataset Information

* Total Samples: **150**
* Features: **4**
* Classes: **3**

---

# 🧠 Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Label Encoding
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Model Comparison
9. Model Saving
10. Streamlit Deployment

---

# 🤖 Models Evaluated

| Model                     | Accuracy    |
| ------------------------- | ----------- |
| K-Nearest Neighbors (KNN) | **100.00%** |
| Logistic Regression       | 96.67%      |
| Support Vector Machine    | 96.67%      |
| Decision Tree             | 93.33%      |
| Random Forest             | 90.00%      |

**Final Selected Model:** K-Nearest Neighbors (KNN)

---

# 📈 Model Performance

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | 100.00% |
| Precision | 100.00% |
| Recall    | 100.00% |
| F1 Score  | 100.00% |

---

# 💻 Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Streamlit
* Plotly
* Pillow
* Pickle

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/PRATIMA1Pratima/codealpha_Iris_Flower_Classification.git
```

Move into the project directory:

```bash
cd Iris_Flower_Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# 🖥️ Application Features

The application allows users to:

* Enter flower measurements
* Predict flower species
* View prediction confidence
* View probability distribution
* Display flower image
* Track prediction history
* Download prediction history as CSV


---

# 📌 Future Improvements

* Deploy on Streamlit Community Cloud
* Add support for multiple prediction models
* Add model explainability
* Improve UI/UX with animations
* Enable batch prediction using CSV uploads

---

# 🎯 Learning Outcomes

This project demonstrates:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Machine Learning classification
* Model evaluation
* Model serialization using Pickle
* Streamlit web application development
* Interactive data visualization
* End-to-end ML project deployment

---

# 👩‍💻 Author

**Pratima**

B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)

Aspiring Data Scientist | Machine Learning Enthusiast

---

# ⭐ Support

If you found this project useful, consider giving the repository a **Star ⭐**.

It helps others discover the project and motivates further improvements.

---

# 📄 License

This project is created for educational and portfolio purposes.
