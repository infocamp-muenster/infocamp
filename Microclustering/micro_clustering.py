# micro_clustering.py
import pandas as pd
from datetime import datetime, timedelta
import time
from river import cluster, feature_extraction
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from Macroclustering.macro_clustering_using_database import main_local
from Infodash.globals import global_lock

data_for_export = []

# Funktionen
from datetime import datetime

def convert_date(date_str):
    # Parse the input date string to a datetime object
    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
    
    # Format the datetime object to the desired output format
    european_format_date_str = dt.strftime('%d.%m.%Y %H:%M')
    
    return european_format_date_str

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


# Funktion die das eigentliche Clustern pro Tweet inkrementell durchführt; tweet_cluster_mapping ist dabei Zuordnung von jedem Tweet zu einem Micro-Cluster
def process_tweets(tweets, vectorizer, clustream, tweet_cluster_mapping, stemmer, nlp, stop_words, micro_cluster_centers):
    for _, tweet in tweets.iterrows():
        processed_tweet = preprocess_tweet(tweet['text'], stemmer, nlp, stop_words)
        features = vectorizer.transform_one(processed_tweet)
        try:
            clustream.learn_one(features)
            cluster_id = clustream.predict_one(features)
            # get center for each micro-cluster
            center = clustream.micro_clusters[cluster_id].center
            micro_cluster_centers[cluster_id] = center


            tweet_cluster_mapping.append({
                'tweet_id': tweet['id_str'],
                'cluster_id': cluster_id,
                'timestamp': str(tweet['created_at'])
            })
        except KeyError as e:
            print(f"4. KeyError bei CluStream.learn_one: {e}, tweet: {tweet['text']}, features: {features}")


# Funktion die das cluster_tweet_data Dataframe nach jedem Zeitintervall updated und sämtliche Kennzahlen berechnet
def transform_to_cluster_tweet_data(tweet_cluster_mapping, cluster_tweet_data, start_time, end_time, micro_cluster_centers):
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
                 'lower_threshold', 'upper_threshold', 'center'])

    # Zählen der Tweets für das aktuelle Zeitintervall und Berechnung des Durchschnitts und der Standardabweichung
    rows_to_add = []
    for cluster_id in unique_clusters:
        tweet_count = df_filtered[df_filtered['cluster_id'] == cluster_id].shape[0]

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
            'center': center
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
                'center': center
            })

    new_cluster_tweet_data = pd.concat([new_cluster_tweet_data, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Kombinieren mit dem bestehenden DataFrame
    cluster_tweet_data = pd.concat([cluster_tweet_data, new_cluster_tweet_data], ignore_index=True)
    return cluster_tweet_data


# Funktion um das Dataframe zum dash Script zu liefern
def get_cluster_tweet_data(db, index):
    df = pd.DataFrame()

    while True:
        if global_lock.acquire(blocking=False):  # Versuche, die Lock zu erwerben, ohne zu blockieren
            try:
                if df.empty:
                    print("Trying to get cluster tweet data")
                    cluster_tweet_data = db.search_get_all(index)
                    # Extracting the '_source' part of each dictionary to create a DataFrame
                    data = [item['_source'] for item in cluster_tweet_data]
                    df = pd.DataFrame(data)
                    print("Successfully retrieved cluster tweet data")
                    return df

            except Exception as e:
                print("Fehler bei der Durchführung der Abfragen auf Elasticsearch:", e)

            finally:
                global_lock.release()  # Sicherstellen, dass die Lock immer freigegeben wird

        else:
            # Wenn die Lock nicht verfügbar ist, warte etwas und versuche es erneut
            print("Lock is busy, waiting...")
            time.sleep(5)

def export_data():
    global data_for_export
    return data_for_export


def main_loop(db, index):
    global all_tweets_from_db
    global data_for_export

    try:
        global_lock.acquire(blocking=True)
        all_tweets_from_db = db.search_get_all(index)
    except Exception as e:
        print("Fehler bei der Durchführung der Abfragen auf Elasticsearch:", e)
    finally:
        global_lock.release()

    # TODO: Implement suitable macro-cluster call
    cluster_iterations = 0

    tweets = pd.DataFrame([hit["_source"] for hit in all_tweets_from_db])
    tweets_selected = tweets[['created_at', 'text', 'id_str']]
    tweets_selected.loc[:, 'created_at'] = pd.to_datetime(tweets_selected['created_at'],
                                                          format='%a %b %d %H:%M:%S %z %Y')
    # Initialisierungen
    start_time, end_time = initialize_time_window(tweets_selected, 'created_at')
    vectorizer = feature_extraction.BagOfWords()
    clustream = cluster.CluStream()
    stop_words = set(stopwords.words('english'))  # TODO: Add stopwords for german and other languages
    nlp = spacy.load('en_core_web_sm')
    stemmer = PorterStemmer()

    # cluster_tweet_data Dataframe initialisieren
    columns = ['cluster_id', 'timestamp', 'tweet_count']
    cluster_tweet_data = pd.DataFrame(columns=columns)

    # Zuordnungsliste Cluster id zu Tweet id
    tweet_cluster_mapping = []

    data_for_export = tweet_cluster_mapping
    
    # Dictionary zum Speichern der Mikro-Cluster-Zentren
    micro_cluster_centers = {}

    # Schleife die jeden Tweet des Zeitintervalls behandelt
    while True:
        tweets = fetch_tweets_in_time_window(tweets_selected, start_time, end_time, 'created_at')
        if not tweets.empty:
            print(f"Tweets von {start_time} bis {end_time}:")
            print(tweets[['created_at', 'text', 'id_str']])
            process_tweets(tweets, vectorizer, clustream, tweet_cluster_mapping, stemmer, nlp, stop_words, micro_cluster_centers)

        # Informationen der Microcluster speichern (Zentrum usw.)

        # Cluster_tweet_data Dataframe nach dem Durchlauf des Zeitintervalls aktualisieren
        cluster_tweet_data = transform_to_cluster_tweet_data(tweet_cluster_mapping, cluster_tweet_data, start_time,
                                                             end_time, micro_cluster_centers)


        # Cluster_tweet_data printen zur Kontrolle
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print("Kontroll-Print:")
        print(cluster_tweet_data)

        # Dataframe in Elasticsearch hochladen
        try:
            global_lock.acquire(blocking=True)
            if db.es.indices.exists(index='cluster_tweet_data'):
                db.es.indices.delete(index='cluster_tweet_data')
            db.upload_df('cluster_tweet_data', cluster_tweet_data)
        except Exception as e:
            print(f"An error occurred during upload: {e}")

        finally:
            global_lock.release()

        if cluster_iterations >= 10:
            main_local(cluster_tweet_data)
            cluster_iterations = 0

        # Zeitintervall erhöhen
        start_time += timedelta(minutes=1)
        end_time += timedelta(minutes=1)

        time.sleep(2)
        cluster_iterations += 1
