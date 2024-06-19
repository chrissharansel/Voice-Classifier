import assemblyai as aai

aai.settings.api_key = "aa06ed65065c4a3ebe74218ff3c2831c"
transcriber = aai.Transcriber()


transcript = transcriber.transcribe(r"C:\Users\Dell\Desktop\CSE109\VOICE CLASSIFICATION\codes\voice7.mp3")

print(transcript.text)