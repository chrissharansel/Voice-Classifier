from flask import Flask, render_template, request, redirect, url_for
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import audioread
import speech_recognition as sr
import requests
import assemblyai as aai

app = Flask(__name__)

# Function to extract features from audio files
def extract_features(file_path):
    audio_data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Load the trained model from file path
model_path = "C:/Users/Dell/Desktop/CSE109/VOICE CLASSIFICATION/train_model.pkl"
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Function to record audio from microphone and predict voice type
def predict_voice_type():
    print("Recording...")
    duration = 5  # Recording duration in seconds
    sample_rate = 22050  # Sample rate
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()
    sf.write('temp_recording.wav', recording, sample_rate)  # Save the recording as a temporary WAV file
    features = extract_features('temp_recording.wav')  # Extract features from the recorded audio
    prediction = classifier.predict([features])[0]  # Predict voice type
    predicted_label = convert_label(prediction)
    print("Predicted Voice Type:", predicted_label)
    
def convert_label(prediction):
    if prediction == 0:
        return "Human Voice"
    elif prediction == 1:
        return "Robot Voice"
    elif prediction == 2:
        return "Converted Voice"
    else:
        return "Unknown"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record_live')
def record_live():
    return render_template('record_live.html')

@app.route('/get_audio')
def get_audio():
    return render_template('get_audio.html')

@app.route('/record_get_voice_type')
def record_get_voice_type():
    predict_voice_type()
    return render_template('record_get_voice_type.html', prediction=predicted_label)

@app.route('/record_get_emotion')
def record_get_emotion():
    # Your emotion analysis code here
    return render_template('record_get_emotion.html')

@app.route('/record_get_text')
def record_get_text():
    # Your text analysis code here
    return render_template('record_get_text.html')

@app.route('/get_audio_get_voice_type')
def get_audio_get_voice_type():
    #PERFECT WORKING
    # Your audio analysis code here
    return render_template('get_audio_get_voice_type.html')

@app.route('/get_audio_get_emotion')
def get_audio_get_emotion():
    # Your emotion analysis code here
    return render_template('get_audio_get_emotion.html')

@app.route('/get_audio_get_text')
def get_audio_get_text():
    # Your text analysis code here
    return render_template('get_audio_get_text.html')

if __name__ == '__main__':
    app.run(debug=True)
