import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import sounddevice as sd
import soundfile as sf
import joblib

# Function to extract features from audio files
def extract_features(file_path):
    audio_data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Load the trained model from file path
model_path = "C:/Users/Dell/Desktop/CSE109/VOICE CLASSIFICATION/train_model.pkl"
classifier = joblib.load(model_path)

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
# Call the function to predict voice type from microphone input
predict_voice_type()
