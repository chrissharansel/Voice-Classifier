import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Function to extract features from audio files
def extract_features(file_path):
    audio_data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Dataset path
dataset_path = "C:/Users/Dell/Desktop/CSE109/VOICE CLASSIFICATION/voices"

# Collecting features and labels
features = []
labels = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav") or file.endswith(".mp3"):
            file_path = os.path.join(root, file)
            label = os.path.basename(root)
            features.append(extract_features(file_path))
            labels.append(label)

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Training the model
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# Predicting on test set
y_pred = classifier.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model to a file
model_filename = "C:/Users/Dell/Desktop/CSE109/VOICE CLASSIFICATION/train_model.pkl"
with open(model_filename, 'wb') as model_file:
    joblib.dump(classifier, model_file)

print("Model saved successfully.")

