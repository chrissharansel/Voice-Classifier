import speech_recognition

recognizer = speech_recognition.Recognizer()

while True:
    try:
        audio_file = input("Enter the path to the audio file (wav or mp3): ")
        with speech_recognition.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            text = text.lower()
            print(f"Recognized: {text}")
            break  # Exit the loop after successful recognition
    except FileNotFoundError:
        print("File not found. Please enter a valid file path.")
        continue
    except speech_recognition.UnknownValueError:
        print("Could not understand audio.")
        continue