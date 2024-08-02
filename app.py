from flask import Flask, request, render_template
from models import model_MDM, model_SCLB, model_NCLB
from PIL import Image
import numpy as np

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or 'model_type' not in request.form:
            return render_template('index.html', result="No file or model type selected")
        
        file = request.files['file']
        model_type = request.form['model_type']
        
        if file.filename == '':
            return render_template('index.html', result="No selected file")
        
        if file:
            image = Image.open(file.stream)
            result = predict_disease(image, model_type)
            return render_template('index.html', result=result)
    
    return render_template('index.html', result=None)

if __name__ == "__main__":
    app.run(debug=True)
