import pandas as pd
from openai import OpenAI
import os

# OpenAI API-Schlüssel konfigurieren
client = OpenAI(api_key='sk-proj-OXHaIyzphXpCX181eu20T3BlbkFJIv1dNOenFbjO9V4zaY3b')

# Tweets einlesen
#df = pd.read_csv('tweets_extracted.csv')
#file_path = os.path.join(os.path.dirname(__file__), 'tweets_extracted.csv')
#df = pd.read_csv(file_path)

# Funktion zur Erstellung einer Zusammenfassung
def summarize_tweets(data):
    # Kombiniere alle Tweets in einen einzigen String
    #combined_tweets = "\n".join(tweets)

    # Anfrage an die OpenAI API, um die Tweets zusammenzufassen
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Determine the single-word topic with a maximum of 12 letters of the following tweets, then set : and briefly explain the topic with the content of the tweets further in english."},
        {"role": "user", "content": data}
    ])

    # Rückgabe der Zusammenfassung
    summary = response.choices[0].message.content.strip()
    return summary
