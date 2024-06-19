
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
def predict_voice_type():
    print("Recording...")
    duration = 5  # Recording duration in seconds
    sample_rate = 22050  # Sample rate
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()
    sf.write('temp_recording.wav', recording, sample_rate)  # Save the recording as a temporary WAV file
    features = extract_features('temp_recording.wav')  # Extract features from the recorded audio
    prediction = clf.predict([features])[0]  # Predict voice type
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

@app.route('/find_voice_type')
def find_voice_type():
    return render_template('find_voice_type.html')

@app.route('/find_emotion')
def find_emotion():
    return render_template('find_emotion.html')

@app.route('/classify_voice', methods=['POST'])
def classify_voice():
    # Get uploaded audio file
    audio_file = request.files['audio_file']
    
    # Save the uploaded file to a temporary location
    temp_path = 'temp_audio.wav'
    audio_file.save(temp_path)
    
    # Extract features from the uploaded audio file
    new_features = extract_features(temp_path)
    
    # Predict the voice type using the trained classifier
    predicted_label = clf.predict([new_features])[0]
    
    # Remove the temporary audio file
    os.remove(temp_path)
    
    return render_template('find_voice_type.html', prediction=predicted_label)

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    # Get uploaded audio file
    audio_file = request.files['audio_file']
    
    # Save the uploaded file to a temporary location
    temp_path = 'temp_audio.wav'
    audio_file.save(temp_path)
    
    # Transcribe audio
    transcript = transcribe_audio(temp_path)
    
    # Analyze sentiment
    if transcript:
        sentiment_result = analyze_sentiment(transcript)
        print("Sentiment Analysis Result:", sentiment_result)
    
    # Remove the temporary audio file
    os.remove(temp_path)
    
    return render_template('find_emotion.html', sentiment=sentiment_result)

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
