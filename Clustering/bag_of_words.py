from river import feature_extraction
from collections import defaultdict
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
from Database import Database

# TODO: Implement function which gets text from tweets
texts = [
    "Das ist ist ist ist ist ist ist ist ist ist ist ist ist ist ist ist ein Beispieltext und der ist auf jeden Fall echt cool (ist so!).",
    "Hier ist noch ein Beispieltext.",
    "Wir konvertieren diesen Text in ein Bag-of-Words-Modell."
]

# Generate bag-of-words model
bow = feature_extraction.BagOfWords(lowercase=True, strip_accents=True)
stop_words = set(stopwords.words('german'))

# Preprocess tweets: Tokenize text & remove stopwords
def preprocess_text(texts, stop_words):
    tokenized_texts = [word_tokenize(text) for text in texts]
    cleaned_tokens = [
        [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
        for tokens in tokenized_texts
    ]
    return cleaned_tokens

processed_texts = preprocess_text(texts, stop_words)

# Transform tweets from one makro cluster in vectors & combine them
combined_vector = defaultdict(int)
for text in processed_texts:
    doc = ' '.join(text)
    vector = bow.transform_one(doc)
    for key, value in vector.items():
        combined_vector[key] += value

word_freq = dict(combined_vector)

# Generating wordcloud & save it as png
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

# TODO: Save png with name of macro cluster
wordcloud.to_file('name_of_makro_cluster.png')