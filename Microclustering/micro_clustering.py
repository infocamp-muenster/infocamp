# micro_clustering.py
import pandas as pd
from datetime import timedelta, datetime
import time
from river import cluster, feature_extraction
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from Macroclustering.macro_clustering_using_database import main_macro
from Infodash.globals import global_lock
from textClustPy import textclust
from textClustPy import Preprocessor
from textClustPy import InMemInput
import pandas as pd
from datetime import datetime


# TODO Macro-Clustering noch implementieren

data_for_export = []

def convert_date(date_str):
    # Parse the input date string to a datetime object
    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

    # Format the datetime object to the desired output format
    european_format_date_str = dt.strftime('%d.%m.%Y %H:%M:%S')

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


# Funktion die das eigentliche Clustern pro Tweet inkrementell durchführt; tweet_cluster_mapping ist dabei Zuordnung
# von jedem Tweet zu einem Micro-Cluster
def process_tweets(tweets, tweet_cluster_mapping, db):
    # cluster_tweet_data Dataframe initialisieren
    columns = ['cluster_id', 'timestamp', 'tweet_count']
    cluster_tweet_data = pd.DataFrame(columns=columns)

    # Initializing macro-cluster call
    macro_cluster_iterations = 3  # Counter after how many micro-clustering iterations macro clustering starts
    micro_cluster_iterations = 0  # Setting micro-cluster iterations initially on 0


    # Configuration of TextClust object and Preprocessor with standard parameters for twitter data
    clust = textclust(radius=0.2, _lambda=0.001, tgap=100, termfading=True, realtimefading=True, num_macro=10,
                      minWeight=0, micro_distance="tfidf_cosine_distance", macro_distance="tfidf_cosine_distance",
                      idf=True, auto_merge=True, auto_r=True, verbose=False)

    preprocessor = Preprocessor(language='english', max_grams=2, stemming=False, hashtag=True, stopword_removal=True,
                                username=True, punctuation=True, url=True)

    # Set column IDs of tweet-IDs, timestamp and text
    clust_input = InMemInput(textclust=clust, preprocessor=preprocessor, pdframe=tweets, col_id=2, col_time=0, col_text=1)

    # Give textClust only tgap observations to be able to store the results in between
    for start in range(0, len(tweets), clust.tgap):
        start_time = tweets.iloc[start]['created_at']
        end_time = tweets.iloc[min(start + clust.tgap, len(tweets) - 1)]['created_at']  # Ensure we don't go out of bounds
        clust_input.update(len(tweets[start:start + clust.tgap]))
        microclusters = clust.microclusters

        for cluster_id, microcluster in microclusters.items():
            for textid in microcluster.textids:
                tweet = tweets.loc[tweets['id_str'] == textid].iloc[0]
                tweet_cluster_mapping.append({
                    'tweet_id': tweet['id_str'],
                    'cluster_id': cluster_id,
                    'timestamp': str(tweet['created_at'])
                })

        cluster_center = []
        cluster_tweet_data = transform_to_cluster_tweet_data(tweet_cluster_mapping, cluster_tweet_data, start_time,
                                                             end_time, cluster_center)

        # Upload dataframe to elasticsearch database
        print(cluster_tweet_data)
        try:
            global_lock.acquire(blocking=True)
            if db.es.indices.exists(index='cluster_tweet_data'):
                db.es.indices.delete(index='cluster_tweet_data')
            db.upload_df('cluster_tweet_data', cluster_tweet_data)
        except Exception as e:
            print(f"An error occurred during upload: {e}")
        finally:
            global_lock.release()

        # Eventually starting macro-clustering with distance-matrix
        if micro_cluster_iterations >= macro_cluster_iterations:
            dm = clust.get_distance_matrix(clust.getmicroclusters())
            main_macro('Textclust', dm)
            micro_cluster_iterations = 0

        # Zeitintervall erhöhen
        start_time += timedelta(minutes=1)
        end_time += timedelta(minutes=1)

        micro_cluster_iterations += 1
        time.sleep(15)

    return cluster_tweet_data


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
        #center = micro_cluster_centers.get(cluster_id, None)
        center = 0

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
            #center = micro_cluster_centers.get(cluster_id, None)
            center = 0

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


def export_data():
    global data_for_export
    return data_for_export


def main_loop(db, index):
    global all_tweets_from_db
    global data_for_export
    
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
    tweets_selected.loc[:, 'created_at'] = pd.to_datetime(tweets_selected['created_at'],
                                                          format='%a %b %d %H:%M:%S %z %Y')

    # Sorting dataframe ascending via 'created_at'
    tweets_selected = tweets_selected.sort_values(by='created_at', ascending=True)

    # Initialisierungen
    #start_time, end_time = initialize_time_window(tweets_selected, 'created_at')
    #vectorizer = feature_extraction.BagOfWords()
    #clustream = cluster.CluStream()
    #stop_words = set(stopwords.words('english'))  # TODO: Add stopwords for german and other languages
    #nlp = spacy.load('en_core_web_sm')
    #stemmer = PorterStemmer()

    # cluster_tweet_data Dataframe initialisieren
    #columns = ['cluster_id', 'timestamp', 'tweet_count']
    #cluster_tweet_data = pd.DataFrame(columns=columns)

    # Zuordnungsliste Cluster id zu Tweet id
    tweet_cluster_mapping = []

    data_for_export = tweet_cluster_mapping

    # Dictionary zum Speichern der Mikro-Cluster-Zentren
    micro_cluster_centers = {}

    # Schleife die jeden Tweet des Zeitintervalls behandelt

    process_tweets(tweets_selected, tweet_cluster_mapping, db)
    '''
    while True:
        tweets = fetch_tweets_in_time_window(tweets_selected, start_time, end_time, 'created_at')
        if not tweets.empty:
            print(f"Tweets von {start_time} bis {end_time}:")
            print(tweets[['created_at', 'text', 'id_str']])
            process_tweets(tweets, vectorizer, clustream, tweet_cluster_mapping, stemmer, nlp, stop_words,
                           micro_cluster_centers)

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
            main_macro()
            micro_cluster_iterations = 0

        # Zeitintervall erhöhen
        start_time += timedelta(minutes=1)
        end_time += timedelta(minutes=1)

        time.sleep(2)
        micro_cluster_iterations += 1
    '''
