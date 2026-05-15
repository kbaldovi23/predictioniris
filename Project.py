import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Iris Species Classification",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Species Classification")
st.write("Machine Learning Project using Random Forest")

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

species_names = iris.target_names

X['species'] = y
X['species_name'] = X['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

# -------------------------------------------------
# DATASET SECTION
# -------------------------------------------------

st.header("📊 Dataset Overview")

st.write("First rows of dataset:")
st.dataframe(X.head())

st.write("Dataset shape:")
st.write("Project developed for Data Mining course")
st.write(X.shape)

# -------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------

features = iris.data
labels = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------------------------
# PREDICTIONS
# -------------------------------------------------

y_pred = model.predict(X_test)

# -------------------------------------------------
# METRICS
# -------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# -------------------------------------------------
# METRICS SECTION
# -------------------------------------------------

st.header("📈 Model Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------

st.header("🧩 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig_cm, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=species_names,
    yticklabels=species_names,
    ax=ax
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig_cm)

# -------------------------------------------------
# HISTOGRAMS
# -------------------------------------------------

st.header("📉 Histograms")

fig_hist = px.histogram(
    X,
    x='sepal length (cm)',
    color='species_name',
    barmode='overlay'
)

st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------------------------
# SCATTER MATRIX
# -------------------------------------------------

st.header("🔍 Scatter Matrix")

fig_scatter = px.scatter_matrix(
    X,
    dimensions=iris.feature_names,
    color='species_name'
)

st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------
# USER INPUT SECTION
# -------------------------------------------------

st.header("🌺 Predict New Flower")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length",
        min_value=0.0,
        max_value=10.0,
        value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width",
        min_value=0.0,
        max_value=10.0,
        value=3.5
    )

with col2:
    petal_length = st.number_input(
        "Petal Length",
        min_value=0.0,
        max_value=10.0,
        value=1.4
    )

    petal_width = st.number_input(
        "Petal Width",
        min_value=0.0,
        max_value=10.0,
        value=0.2
    )

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------

if st.button("Predict Species"):

    input_data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    prediction = model.predict(input_data)

    predicted_species = species_names[prediction[0]]

    st.success(f"Predicted Species: {predicted_species}")

    # ---------------------------------------------
    # 3D SCATTER PLOT
    # ---------------------------------------------

    st.header("🌐 3D Scatter Plot")

    fig_3d = px.scatter_3d(
        X,
        x='sepal length (cm)',
        y='sepal width (cm)',
        z='petal length (cm)',
        color='species_name',
        opacity=0.7
    )

    fig_3d.add_scatter3d(
        x=[sepal_length],
        y=[sepal_width],
        z=[petal_length],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='New Flower'
    )

    st.plotly_chart(fig_3d, use_container_width=True)

# -------------------------------------------------
# CLASSIFICATION REPORT
# -------------------------------------------------

st.header("📄 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=species_names
)

st.text(report)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")
st.write("Project developed for Data Mining course")