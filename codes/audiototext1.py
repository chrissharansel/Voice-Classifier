'''
import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Load the audio file
audio_file = "audio (1).wav"

# Use the recognizer to listen to the audio file and convert it to text
with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)  # Read the entire audio file
    text = recognizer.recognize_google(audio_data)

# Print the transcribed text
print("Transcribed Text:", text)

'''
# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)

