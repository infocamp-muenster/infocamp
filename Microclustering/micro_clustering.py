# micro_clustering.py
import pandas as pd
from datetime import timedelta
import time
from river import cluster, feature_extraction
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from Macroclustering.macro_clustering_using_database import main_macro
from Infodash.globals import global_lock
from Microclustering.textclust import process_tweets_textclust
from Microclustering.detector import Detector

data_for_export = []
all_tweets = []

# Funktionen
from datetime import datetime


def initialize_time_window(df, time_column):
    """
    Diese Funktion initialisiert das Start- und Endzeitfenster basierend auf dem frühesten Zeitstempel.
    """
    start_time = df[time_column].min().floor('min')
    end_time = start_time + timedelta(minutes=1)
    return start_time, end_time


def fetch_tweets_in_time_window(df, start_time, end_time, time_column):
    mask = (df[time_column] >= start_time) & (df[time_column] < end_time)
    return df[mask]


def preprocess_tweet(tweet, stemmer, nlp, stop_words):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\W', ' ', tweet)
    doc = nlp(tweet)
    tokens = [token for token in doc if token.text.lower() not in stop_words and not token.is_punct]
    lemmatized_tokens = [token.lemma_ for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)


# Funktion die das eigentliche Clustern pro Tweet inkrementell durchführt; tweet_cluster_mapping ist dabei Zuordnung
# von jedem Tweet zu einem Micro-Cluster
def process_tweets(tweets, vectorizer, clustream, tweet_cluster_mapping, stemmer, nlp, stop_words,
                   micro_cluster_centers, ai_detector):
    for _, tweet in tweets.iterrows():
        processed_tweet = preprocess_tweet(tweet['text'], stemmer, nlp, stop_words)
        features = vectorizer.transform_one(processed_tweet)
        try:
            clustream.learn_one(features)
            cluster_id = clustream.predict_one(features)

            # AI detector
            result = ai_detector.evaluate("SNNEval", [tweet['text']])
            ai_score = 0
            if result[0] > 0.99:
                ai_score = 1

            # get center for each micro-cluster
            center = clustream.micro_clusters[cluster_id].center
            micro_cluster_centers[cluster_id] = center

            tweet_cluster_mapping.append({
                'tweet_id': tweet['id_str'],
                'cluster_id': cluster_id,
                'timestamp': str(tweet['created_at']),
                'ai_generated': ai_score
            })
        except KeyError as e:
            pass
            # print(f"4. KeyError bei CluStream.learn_one: {e}, tweet: {tweet['text']}, features: {features}")


# Funktion die das cluster_tweet_data Dataframe nach jedem Zeitintervall updated und sämtliche Kennzahlen berechnet
def transform_to_cluster_tweet_data(tweet_cluster_mapping, cluster_tweet_data, start_time, end_time,
                                    micro_cluster_centers):
    """
    Diese Funktion transformiert die tweet_cluster_mapping (Update nach jedem tweet) Liste in eine
    Liste mit sieben Spalten: cluster_id, timestamp, Anzahl der Tweets, durchschnittlicher tweet_count,
    Standardabweichung der tweet_count-Werte, lower_threshold und upper_threshold.
    """
    df = pd.DataFrame(tweet_cluster_mapping)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.floor('min')  # Auf Minutenebene runden

    # Filtern der Daten nach dem gegebenen Zeitintervall
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] < end_time)
    df_filtered = df[mask]

    # Alle einzigartigen Cluster-IDs finden
    unique_clusters = df['cluster_id'].unique()
    previous_clusters = cluster_tweet_data['cluster_id'].unique() if not cluster_tweet_data.empty else []

    # Erstellen einer neuen DataFrame für das aktuelle Zeitintervall
    new_cluster_tweet_data = pd.DataFrame(
        columns=['cluster_id', 'timestamp', 'tweet_count', 'average_tweet_count', 'std_dev_tweet_count',
                 'lower_threshold', 'upper_threshold', 'center', 'ai_abs'])

    # Zählen der Tweets für das aktuelle Zeitintervall und Berechnung des Durchschnitts und der Standardabweichung
    rows_to_add = []
    for cluster_id in unique_clusters:
        tweet_count = df_filtered[df_filtered['cluster_id'] == cluster_id].shape[0]
        ai_abs = df_filtered[(df_filtered['cluster_id'] == cluster_id) & (df_filtered['ai_generated'] == 1).shape[0]]

        # Berechnung des Durchschnitts der bisherigen tweet_count-Werte für diesen Cluster
        previous_counts = cluster_tweet_data[cluster_tweet_data['cluster_id'] == cluster_id][
            'tweet_count'] if not cluster_tweet_data.empty else pd.Series(dtype=float)
        if not previous_counts.empty:
            average_tweet_count = (previous_counts.sum() + tweet_count) / (previous_counts.count() + 1)
            std_dev_tweet_count = np.std(pd.concat([previous_counts, pd.Series([tweet_count])]), ddof=0)
        else:
            average_tweet_count = tweet_count
            std_dev_tweet_count = 0

        # Berechnung der Thresholds
        prev_std_dev_tweet_count = \
            cluster_tweet_data[cluster_tweet_data['cluster_id'] == cluster_id]['std_dev_tweet_count'].iloc[-1] if not \
                cluster_tweet_data[cluster_tweet_data['cluster_id'] == cluster_id].empty else 0
        lower_threshold = tweet_count - 6 * prev_std_dev_tweet_count
        upper_threshold = tweet_count + 6 * prev_std_dev_tweet_count

        # Hinzufügen des Clusterzentrums
        center = micro_cluster_centers.get(cluster_id, None)

        rows_to_add.append({
            'cluster_id': cluster_id,
            'timestamp': end_time,
            'tweet_count': tweet_count,
            'average_tweet_count': average_tweet_count,
            'std_dev_tweet_count': std_dev_tweet_count,
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold,
            'center': center,
            'ai_abs': ai_abs
        })

    # Sicherstellen, dass Cluster ohne Einträge im Zeitintervall hinzugefügt werden
    all_clusters = set(unique_clusters).union(previous_clusters)
    for cluster_id in all_clusters:
        if cluster_id not in [row['cluster_id'] for row in rows_to_add]:
            previous_counts = cluster_tweet_data[cluster_tweet_data['cluster_id'] == cluster_id][
                'tweet_count'] if not cluster_tweet_data.empty else pd.Series(dtype=float)

            # Berechnung des Durchschnitts und der Standardabweichung unter Einbeziehung der 0 für den aktuellen Zeitraum
            if not previous_counts.empty:
                average_tweet_count = (previous_counts.sum()) / (previous_counts.count() + 1)
                std_dev_tweet_count = np.std(pd.concat([previous_counts, pd.Series([0])]), ddof=0)
            else:
                average_tweet_count = 0
                std_dev_tweet_count = 0

            # Berechnung der Thresholds
            prev_std_dev_tweet_count = \
                cluster_tweet_data[cluster_tweet_data['cluster_id'] == cluster_id]['std_dev_tweet_count'].iloc[
                    -1] if not \
                    cluster_tweet_data[cluster_tweet_data['cluster_id'] == cluster_id].empty else 0
            lower_threshold = 0 - 6 * prev_std_dev_tweet_count
            upper_threshold = 0 + 6 * prev_std_dev_tweet_count

            # Hinzufügen des Clusterzentrums
            center = micro_cluster_centers.get(cluster_id, None)

            rows_to_add.append({
                'cluster_id': cluster_id,
                'timestamp': end_time,
                'tweet_count': 0,
                'average_tweet_count': average_tweet_count,
                'std_dev_tweet_count': std_dev_tweet_count,
                'lower_threshold': lower_threshold,
                'upper_threshold': upper_threshold,
                'center': center,
                'ai_abs': 0
            })

    new_cluster_tweet_data = pd.concat([new_cluster_tweet_data, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Kombinieren mit dem bestehenden DataFrame
    cluster_tweet_data = pd.concat([cluster_tweet_data, new_cluster_tweet_data], ignore_index=True)
    print("AI DF")
    print(cluster_tweet_data)
    return cluster_tweet_data


def export_data():
    global data_for_export
    global all_tweets

    # Daten in DataFrames umwandeln
    mapping = pd.DataFrame(data_for_export)
    tweets = pd.DataFrame(all_tweets)

    if tweets.empty:
        return mapping
    
    if mapping.empty:
        return tweets   

    if not mapping.empty and not tweets.empty:
         # Anpassungen für den join
        tweets = tweets.rename(columns={'id_str':'tweet_id'})
        tweets = tweets.drop(columns=['created_at'])

        # Join der beiden DataFrames
        result = pd.merge(mapping, tweets, how="left", on="tweet_id")
        return result


def main_loop(db, index, micro_algo):
    global all_tweets_from_db
    global data_for_export
    global all_tweets
    

    print("Starting micro_clustering main loop...")

    try:
        global_lock.acquire(blocking=True)
        all_tweets_from_db = db.search_get_all(index)
    except Exception as e:
        print("Fehler bei der Durchführung der Abfragen auf Elasticsearch:", e)
    finally:
        global_lock.release()



    # Initializing macro-cluster call
    macro_cluster_iterations = 8  # Counter after how many micro-clustering iterations macro clustering starts
    micro_cluster_iterations = 0  # Setting micro-cluster iterations initially on 0

    tweets = pd.DataFrame([hit["_source"] for hit in all_tweets_from_db])
    tweets_selected = tweets[['created_at', 'text', 'id_str']]
    all_tweets = tweets_selected
    tweets_selected.loc[:, 'created_at'] = pd.to_datetime(tweets_selected['created_at'],
                                                          format='%a %b %d %H:%M:%S %z %Y')

    # Zuordnungsliste Cluster id zu Tweet id
    tweet_cluster_mapping = []

    data_for_export = tweet_cluster_mapping

    if micro_algo == "Textclust":
        # Sorting dataframe ascending via 'created_at'
        tweets_selected = tweets_selected.sort_values(by='created_at', ascending=True)
        process_tweets_textclust(tweets_selected, tweet_cluster_mapping, db)

    if micro_algo == "Clustream":

        # Initialisierungen
        start_time, end_time = initialize_time_window(tweets_selected, 'created_at')
        vectorizer = feature_extraction.BagOfWords()
        clustream = cluster.CluStream()
        stop_words = set(stopwords.words('english'))  # TODO: Add stopwords for german and other languages
        nlp = spacy.load('en_core_web_sm')
        stemmer = PorterStemmer()
        ai_detector = Detector('http://ls-stat-ml.uni-muenster.de:7100/compute')


        # cluster_tweet_data Dataframe initialisieren
        columns = ['cluster_id', 'timestamp', 'tweet_count']
        cluster_tweet_data = pd.DataFrame(columns=columns)

        # Initializing macro-cluster call
        macro_cluster_iterations = 8  # Counter after how many micro-clustering iterations macro clustering starts
        micro_cluster_iterations = 0  # Setting micro-cluster iterations initially on 0

        # Dictionary zum Speichern der Mikro-Cluster-Zentren
        micro_cluster_centers = {}

        # Schleife die jeden Tweet des Zeitintervalls behandelt
        while True:
            tweets = fetch_tweets_in_time_window(tweets_selected, start_time, end_time, 'created_at')
            if not tweets.empty:
                print(f"Process tweets from {start_time} to {end_time}:")
                # print(tweets[['created_at', 'text', 'id_str']])
                process_tweets(tweets, vectorizer, clustream, tweet_cluster_mapping, stemmer, nlp, stop_words,
                               micro_cluster_centers, ai_detector)

            # Informationen der Microcluster speichern (Zentrum usw.)

            # Cluster_tweet_data Dataframe nach dem Durchlauf des Zeitintervalls aktualisieren
            cluster_tweet_data = transform_to_cluster_tweet_data(tweet_cluster_mapping, cluster_tweet_data, start_time,
                                                                 end_time, micro_cluster_centers)

            # Cluster_tweet_data printen zur Kontrolle
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print("Control-Print cluster_tweet_data got transformed successfully and is ready for upload!")
            # print(cluster_tweet_data)

            # Upload dataframe to elasticsearch database
            try:
                global_lock.acquire(blocking=True)
                if db.es.indices.exists(index='cluster_tweet_data'):
                    db.es.indices.delete(index='cluster_tweet_data')
                db.upload_df('cluster_tweet_data', cluster_tweet_data)
            except Exception as e:
                print(f"An error occurred during upload: {e}")
            finally:
                global_lock.release()

            # Eventually starting macro-clustering
            if micro_cluster_iterations >= macro_cluster_iterations:
                main_macro("Clustream")
                micro_cluster_iterations = 0

            # Zeitintervall erhöhen
            start_time += timedelta(minutes=1)
            end_time += timedelta(minutes=1)

            time.sleep(2)
            micro_cluster_iterations += 1
