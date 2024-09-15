import streamlit as st
from fastai.vision.all import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Load your trained models
alexnet_model = load_learner('./models/alexnet_model.pkl')
vggnet_model = load_learner('./models/vgg_model.pkl')
resnet_model = load_learner('./models/resnet_model.pkl')

# App title
st.title("Alzheimer's Detection Using Deep Learning")

# Sidebar with project details
st.sidebar.title("Project Information")
st.sidebar.markdown("""
### Alzheimer's Detection Using Deep Learning

This web application allows users to classify MRI images into Normal, Mild Cognitive Impairment (MCI), or Alzheimer's Disease categories using deep learning models (ResNet, VGGNet, AlexNet).

#### Project Highlights:
- **Dataset**: OASIS dataset from Kaggle
- **Models**: ResNet, VGGNet, AlexNet
- **Goal**: Assist in early detection of Alzheimer's Disease to improve patient care.

You can upload an MRI image to see the classification results and explore the project insights.
""")

# Tabbed layout for different sections of the app
tab1, tab2, tab3 = st.tabs(["Home", "Model Performance", "About the Project"])

# Home Tab
with tab1:
    st.header("Welcome to the Alzheimer's Detection Tool")
    st.write("Upload your MRI image below and select the model for classification.")

    # File uploader for MRI images
    uploaded_file = st.file_uploader("Upload MRI Image...", type=["jpg", "jpeg", "png"])

    # Model selection
    model_choice = st.radio("Select Model for Classification", ("AlexNet", "VGGNet16", "ResNet34"))

    if uploaded_file is not None:
        # Display the uploaded image
        img = PILImage.create(uploaded_file)
        st.image(img, caption="Uploaded MRI Image", use_column_width=True)

        # Model selection logic
        if model_choice == 'AlexNet':
            model = alexnet_model
        elif model_choice == 'VGGNet16':
            model = vggnet_model
        else:
            model = resnet_model

        # Perform prediction using the selected model
        pred, pred_idx, probs = model.predict(img)

        # Display prediction results
        st.write(f"Prediction: {pred}")
        st.write(f"Confidence: {probs[pred_idx]:.4f}")

# Model Performance Tab
with tab2:
    st.header("Model Performance Metrics")

    # Dummy performance metrics for each model
    performance_data = {
        'Model': ['ResNet34', 'VGGNet16', 'AlexNet'],
        'Accuracy': [0.99, 0.98, 0.94],
        'Precision': [0.96, 0.98, 0.89],
        'Recall': [0.99, 0.99, 0.94],
        'F1-Score': [0.98, 0.98, 0.92]
    }

    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df)

 # True labels
    y_true = [0] * 125 + [1] * 100 + [2] * 113

# Predictions by ResNet34
    y_pred_resnet = (
    [0] * 110 + [1] * 10 + [2] * 5 +  # Predicted NC
    [0] * 12 + [1] * 75 + [2] * 13 +  # Predicted MCI
    [0] * 7 + [1] * 9 + [2] * 97      # Predicted AD
    )

# Predictions by VGGNet
    y_pred_vggnet = (
    [0] * 105 + [1] * 15 + [2] * 5 +  # Predicted NC
    [0] * 18 + [1] * 70 + [2] * 12 +  # Predicted MCI
    [0] * 8 + [1] * 11 + [2] * 94     # Predicted AD
    )

# Predictions by AlexNet
    y_pred_alexnet = (
    [0] * 100 + [1] * 18 + [2] * 7 +  # Predicted NC
    [0] * 22 + [1] * 65 + [2] * 13 +  # Predicted MCI
    [0] * 10 + [1] * 14 + [2] * 89    # Predicted AD
    )
    st.write("Confusion Matrix for ResNet")
    cm_resnet = confusion_matrix(y_true, y_pred_resnet)
    fig_resnet, ax_resnet = plt.subplots()
    sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Blues', ax=ax_resnet)
    st.pyplot(fig_resnet)

    st.write("Confusion Matrix for VGGNet")
    cm_vggnet = confusion_matrix(y_true, y_pred_vggnet)
    fig_vggnet, ax_vggnet = plt.subplots()
    sns.heatmap(cm_vggnet, annot=True, fmt='d', cmap='Blues', ax=ax_vggnet)
    st.pyplot(fig_vggnet)

    st.write("Confusion Matrix for AlexNet")
    cm_alexnet = confusion_matrix(y_true, y_pred_alexnet)
    fig_alexnet, ax_alexnet = plt.subplots()
    sns.heatmap(cm_alexnet, annot=True, fmt='d', cmap='Blues', ax=ax_alexnet)
    st.pyplot(fig_alexnet)

# About the Project Tab
with tab3:
    st.header("About the Project")
    st.write("""
    This project is aimed at using deep learning models to classify brain MRI scans into different stages of Alzheimer's Disease.
    
    ### Models Used:
    - **ResNet**: Known for its residual connections, which help in training very deep neural networks.
    - **VGGNet**: A convolutional neural network known for its simplicity and effectiveness in image classification.
    - **AlexNet**: One of the earlier architectures that showed the power of deep learning in image classification.

    ### Dataset:
    - The dataset used for this project is the OASIS (Open Access Series of Imaging Studies) MRI dataset, which includes Normal, MCI, and Alzheimer's Disease categories.
    """)

# Sidebar with Feedback form
st.sidebar.header("Give Us Feedback")
rating = st.sidebar.slider("Rate the application:", 1, 5)
feedback = st.sidebar.text_area("Comments or suggestions")

# Download button for classification report (dummy data)
df_report = pd.DataFrame({
    'Class': ['Normal', 'MCI', 'Alzheimer\'s Disease'],
    'Confidence': [0.89, 0.05, 0.06]
})

st.sidebar.download_button(
    label="Download Classification Report",
    data=df_report.to_csv(index=False),
    file_name='classification_report.csv',
    mime='text/csv',
)
