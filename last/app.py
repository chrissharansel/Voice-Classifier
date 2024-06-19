from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sounddevice as sd
import soundfile as sf
import joblib
import speech_recognition as sr
import requests
import audioread
import assemblyai as aai
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


app = Flask(__name__)
recognizer = sr.Recognizer()
aai.settings.api_key = "aa06ed65065c4a3ebe74218ff3c2831c"
transcriber = aai.Transcriber()
# Function to recognize speech from the microphone
def recognize_speech():
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

# Function to extract features from audio files
def extract_features_audio(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    except audioread.NoBackendError:
        raise Exception("No audio backend available. Please install ffmpeg or use WAV files.")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    return mfccs_processed
def extract_features(file_path):
    audio_data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean
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
            features = extract_features_audio(file_path)
            X.append(features)
            # Append label to y
            y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)


# Load the trained model
model_path = "C:/Users/Dell/Desktop/CSE109/VOICE CLASSIFICATION/last/train_model.pkl"
classifier = joblib.load(model_path)

# Function to predict voice type from live audio
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
    return predicted_label

# Function to convert label indices to voice type labels
def convert_label(prediction):
    if prediction == 0:
        return "Human Voice"
    elif prediction == 1:
        return "Robot Voice"
    elif prediction == 2:
        return "Converted Voice"
    else:
        return "Unknown"
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
    features = extract_features_audio(audio_file)
    # Predict the voice type
    predicted_label = clf.predict([features])[0]
    return predicted_label

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the record_live page
@app.route('/record_live')
def record_live():
    return render_template('record_live.html')

@app.route('/get_audio_upload')
def get_audio_upload():
    return render_template('get_audio_upload.html')

# Route to get voice type from live audio
@app.route('/get_voice_type')
def get_voice_type():
    predicted_label = predict_voice_type()
    return jsonify({"voice_type": predicted_label})

# Route to get emotion from live audio
@app.route('/get_emotion_record')
def get_emotion_record():
    text = recognize_speech()
    if text:
        sentiment_result = analyze_sentiment(text)
        return jsonify(sentiment_result)
    else:
        return "Could not understand audio"

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
