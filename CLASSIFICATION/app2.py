
from flask import Flask, render_template, request
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import audioread
import requests
import assemblyai as aai
import speech_recognition as sr

app = Flask(__name__)

# Function to extract features from audio file
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    except audioread.NoBackendError:
        raise Exception("No audio backend available. Please install ffmpeg or use WAV files.")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Function to transcribe audio from an mp3 file
def transcribe_audio(file_path):
    aai.settings.api_key = "aa06ed65065c4a3ebe74218ff3c2831c"
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(file_path)
    return transcript.text.lower()

# Function to recognize speech from the microphone
def recognize_speech():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as mic:
        print("Listening...")
        audio = recognizer.listen(mic)

    try:
        text = recognizer.recognize_google(audio)
        text = text.lower()
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None

# Function to analyze text sentiment using the sentiment analysis API
def analyze_sentiment(text):
    url = "https://text-sentiment-analyzer-api1.p.rapidapi.com/sentiment"
    payload = {"text": text}
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "X-RapidAPI-Key": "f956d5273fmsh72cb93108ac9af8p191aafjsnfb53d5699775",
        "X-RapidAPI-Host": "text-sentiment-analyzer-api1.p.rapidapi.com"
    }

    response = requests.post(url, data=payload, headers=headers)
    return response.json()

# Function to predict voice type
def predict_voice_type(file_path):
    features = extract_features(file_path)  # Extract features from the recorded audio
    prediction = clf.predict([features])[0]  # Predict voice type
    predicted_label = convert_label(prediction)
    return predicted_label

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

@app.route('/record_find_voice_type', methods=['POST'])
def record_find_voice_type():
    audio_file = request.files['audio_file']
    temp_path = 'temp_audio.wav'
    audio_file.save(temp_path)
    predicted_label = predict_voice_type(temp_path)
    os.remove(temp_path)
    return render_template('record_find_voice_type.html', prediction=predicted_label)

@app.route('/record_find_emotion', methods=['POST'])
def record_find_emotion():
    # Handle emotion analysis for recorded audio
    return render_template('record_find_emotion.html')

@app.route('/record_find_text', methods=['POST'])
def record_find_text():
    # Handle text analysis for recorded audio
    return render_template('record_find_text.html')

@app.route('/get_audio')
def get_audio():
    return render_template('get_audio.html')

@app.route('/get_find_voice_type', methods=['POST'])
def get_find_voice_type():
    audio_file = request.files['audio_file']
    temp_path = 'temp_audio.wav'
    audio_file.save(temp_path)
    predicted_label = predict_voice_type(temp_path)
    os.remove(temp_path)
    return render_template('get_find_voice_type.html', prediction=predicted_label)

@app.route('/get_find_emotion', methods=['POST'])
def get_find_emotion():
    # Handle emotion analysis for uploaded audio
    return render_template('get_find_emotion.html')

@app.route('/get_find_text', methods=['POST'])
def get_find_text():
    # Handle text analysis for uploaded audio
    return render_template('get_find_text.html')

if __name__ == '__main__':
    # Path to the folder containing audio files
    base_path = r'C:\Users\Dell\Desktop\CSE109\VOICE CLASSIFICATION\voices'

    # List of labels
    labels = ['convertedvoice', 'humanvoice', 'robotvoice']

    # Initialize lists to store features and labels
    X = []
    y = []

    # Iterate over each label
    for label in labels:
        # Path to the folder for the current label
        folder_path = os.path.join(base_path, label)
        # Iterate over each audio file in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is an audio file
            if filename.endswith(('.wav', '.mp3')):
                # Full path to the audio file
                file_path = os.path.join(folder_path, filename)
                # Extract features and append to X
                features = extract_features(file_path)
                X.append(features)
                # Append label to y
                y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Train the Random Forest classifier
    clf.fit(X, y)

    app.run(debug=True)