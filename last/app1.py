from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import audioread
from flask import Flask, render_template, request, jsonify
import assemblyai as aai
import requests
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import requests

@app.route('/')
def index():
    return render_template('index.html')
app = Flask(__name__)
recognizer = sr.Recognizer()
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




# Route to get emotion from live audio
@app.route('/get_emotion_record')
def get_emotion_record():
    text = recognize_speech()
    if text:
        sentiment_result = analyze_sentiment(text)
        return jsonify(sentiment_result)
    else:
        return "Could not understand audio"

@app.route('/record_live')
def record_live():
    return render_template('record_live.html')

# Route to get text from live audio
@app.route('/get_text')
def get_text():
    try:
        with sr.Microphone() as mic:
            print("Listening...")
            audio = recognizer.listen(mic)
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            text = text.lower()
            print(f"Recognized: {text}")
            return text
    except sr.UnknownValueError:
        return "Could not understand audio"

# Function to extract features from audio file
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    except audioread.NoBackendError:
        raise Exception("No audio backend available. Please install ffmpeg or use WAV files.")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    return mfccs_processed

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

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Route for the HTML page
@app.route('/get_audio_upload')
def get_audio_upload():
    return render_template('get_audio_upload.html')

# Route to handle voice type prediction
@app.route('/predict_voice_type_audio', methods=['POST'])
def predict_voice_type_audio():
    # Check if the request contains a file
    if 'audio' not in request.files:
        return "No audio file found"

    audio_file = request.files['audio']
    # Check if the file is empty
    if audio_file.filename == '':
        return "No selected file"

    # Extract features from the audio file
    features = extract_features(audio_file)
    # Predict the voice type
    predicted_label = clf.predict([features])[0]
    return predicted_label

# Configure AssemblyAI settings
aai.settings.api_key = "aa06ed65065c4a3ebe74218ff3c2831c"
transcriber = aai.Transcriber()



# Route to handle audio transcription
@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    # Check if the request contains a file
    if 'audio' not in request.files:
        return "No audio file found"

    audio_file = request.files['audio']
    # Check if the file is empty
    if audio_file.filename == '':
        return "No selected file"

    # Transcribe the audio
    transcript = transcribe_audio(audio_file)
    return transcript

# Function to transcribe audio from an mp3 file
def transcribe_audio(audio_file):
    transcript = transcriber.transcribe(audio_file)
    if isinstance(transcript, str):
        # If the transcript is a string, return it directly
        return transcript.lower()
    else:
        # If the transcript is an object, return its text attribute
        return transcript.text.lower()

# Route to get emotion from transcribed text
@app.route('/get_emotion', methods=['POST'])
def get_emotion():
    # Check if the request contains a file
    if 'audio' not in request.files:
        return "No audio file found"

    audio_file = request.files['audio']
    # Check if the file is empty
    if audio_file.filename == '':
        return "No selected file"

    # Transcribe the audio
    transcript = transcribe_audio(audio_file)

    if transcript:
        # Analyze text sentiment using the sentiment analysis API
        url = "https://text-sentiment-analyzer-api1.p.rapidapi.com/sentiment"
        payload = {"text": transcript}
        headers = {
            "content-type": "application/x-www-form-urlencoded",
            "X-RapidAPI-Key": "f956d5273fmsh72cb93108ac9af8p191aafjsnfb53d5699775",
            "X-RapidAPI-Host": "text-sentiment-analyzer-api1.p.rapidapi.com"
        }
        response = requests.post(url, data=payload, headers=headers)
        return jsonify(response.json())
    else:
        return "No text provided"


if __name__ == '__main__':
    app.run(debug=True)
