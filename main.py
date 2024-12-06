import tensorflow as tf
from keras.models import load_model 
import os
import streamlit as st
import librosa
import numpy as np
st.title("Emotion Detection Through Speech/Audio")

if not os.path.exists("uploads"):
    os.makedirs("uploads")

def save_file(audio_file):
    try:
        file_path = os.path.join("uploads", audio_file.name)
        with open(file_path, 'wb') as f:
            f.write(audio_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_feature(filename):
    try:
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


audio_file = st.file_uploader("Choose file", type=['wav', 'mp3'])
model = load_model('sp.h5')

predict = st.button("Classify")
if audio_file is not None:
    file_path = save_file(audio_file)
    if file_path:
        st.audio(audio_file)
        # Preprocess and predict
        features = extract_feature(file_path)
        features = [x for x in features]
        features = np.array(features)
        features = np.expand_dims(features, -1)
        features = features.reshape((1,40,1))
        if features is not None and predict:
            try:
                prediction = model.predict(features)# Ensure features are wrapped in a list
                print(prediction) 
                index = np.argmax(prediction)
                predictions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Pleasant Surprised', 'Sad']
                st.title(f"Predicted Emotion: {predictions[index]}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
