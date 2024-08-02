from tensorflow.keras.models import load_model

# Load models
model_MDM = load_model('healthyVSmdm.h5')
model_SCLB = load_model('healthyVSnclb.h5')
model_NCLB = load_model('healthyVSsclb.h5')
