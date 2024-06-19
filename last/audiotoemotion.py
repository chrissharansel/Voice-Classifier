import requests
import assemblyai as aai

# Function to transcribe audio from an mp3 file
def transcribe_audio(file_path):
    aai.settings.api_key = "aa06ed65065c4a3ebe74218ff3c2831c"
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(file_path)
    return transcript.text.lower()

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

# Main function to transcribe audio and analyze sentiment
def main():
    file_path = input("ENter the path to audio:")
    transcript = transcribe_audio(file_path)
    print("Transcription:", transcript)
    
    if transcript:
        sentiment_result = analyze_sentiment(transcript)
        print("Sentiment Analysis Result:", sentiment_result)

if __name__ == "__main__":
    main()
