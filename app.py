import streamlit as st
import requests
from PIL import Image
import numpy as np
from models import model_MDM, model_SCLB, model_NCLB

# Prediction function
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_disease(image, model_type):
    preprocessed_image = preprocess_image(image, (224, 224))  # Adjust size if needed

    if model_type == 'MDM':
        prediction = model_MDM.predict(preprocessed_image)[0][0]
    elif model_type == 'SCLB':
        prediction = model_SCLB.predict(preprocessed_image)[0][0]
    elif model_type == 'NCLB':
        prediction = model_NCLB.predict(preprocessed_image)[0][0]

    # Assuming the model outputs 1 for diseased and 0 for healthy
    if prediction < 0.5:
        return "Healthy"
    else:
        return model_type

# Streamlit app
def main():
    st.title('Maize Leaf Disease Classification')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        model_type = st.selectbox('Select Model Type', ['MDM', 'SCLB', 'NCLB'])

        if st.button('Classify'):
            result = predict_disease(image, model_type)
            st.success(f'This leaf is classified as: {result}')

if __name__ == '__main__':
    main()
