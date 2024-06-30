import time
import pandas as pd
from river import stats
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from Datamanagement.Database import Database
import ast
#from Microclustering.micro_clustering import get_cluster_tweet_data


def get_micro_clusters_id_and_center(db, micro_clusters):
    # TODO: Hier noch Datenbankabfrage implementieren; Diese eventuell noch auslagern in Database File??? (get_cluster_tweet_data aus micro_clustering.py wäre passende Methode)
    # Datenbankabfrage
    #micro_clusters = get_cluster_tweet_data(db, 'cluster_tweet_data')

    #for micro_cluster in micro_clusters:
    #    print(micro_cluster["id"])
    #    print(micro_cluster["center"])

    return micro_clusters


def macro_clustering(micro_clusters):

    # Only apply macro-clustering in case there are any micro-clusters
    if len(micro_clusters) > 0:
        # Use only the last timestamp for macro-clustering
        latest_timestamp = micro_clusters['timestamp'].max()

        # Filter micro-clusters from last timestamp
        latest_micro_clusters = micro_clusters[micro_clusters['timestamp'] == latest_timestamp]

        # Collect all features from the micro-cluster-center
        all_features = set()
        for _, row in latest_micro_clusters.iterrows():
            center = row["center"]
            if isinstance(center, str):
                center = ast.literal_eval(center)
            all_features.update(center.keys())

        # Extract micro-cluster-centers and bring them in generic layout
        micro_cluster_centers = []
        for _, row in latest_micro_clusters.iterrows():
            center = row["center"]
            if isinstance(center, str):
                center = ast.literal_eval(center)
            center_vector = [center.get(feature, 0) for feature in all_features]
            micro_cluster_centers.append(center_vector)

        # Convert list into NumPy-Array
        micro_cluster_centers = np.array(micro_cluster_centers)

        # Initialize KMeans from Scikit-learn and train it on the micro-cluster centers
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(micro_cluster_centers)

        # Predict the macro-clusters
        macro_clusters = kmeans.predict(micro_cluster_centers)
        print(f"Macro clusters: {macro_clusters}")

        # Create a dictionary to store macro-clusters and their belonging micro-clusters
        macro_micro_dict = {}
        for macro_cluster, (_, row) in zip(macro_clusters, latest_micro_clusters.iterrows()):
            if macro_cluster not in macro_micro_dict:
                macro_micro_dict[macro_cluster] = []
            macro_micro_dict[macro_cluster].append(row.to_dict())

        for macro_cluster, micro_cluster in zip(macro_clusters, latest_micro_clusters.iterrows()):
            print(f"Micro-cluster ID: {micro_cluster[1]['cluster_id']}, Macro-cluster ID: {macro_cluster}")

        print(f"Macro micro dict: {macro_micro_dict}")

        return macro_micro_dict


def store_macro_micro_dict_in_database(db, macro_micro_dict):
    # Datenbankanfrage
    # TODO: Not tested yet, only tested on local base
    try:
        if db.es.indices.exists(index='macro_micro_dict'):
            db.es.indices.delete(index='macro_micro_dict')
        db.upload_df('macro_micro_dict', macro_micro_dict)
    except Exception as e:
        print(f"An error occurred during upload: {e}")
    return

# TODO: Not tested yet
def get_macro_micro_dict_from_database(db, index):
    try:
        micro_macro_dict = db.search_get_all(index)
        # Extracting the '_source' part of each dictionary to create a DataFrame
        data = [item['_source'] for item in micro_macro_dict]
        df = pd.DataFrame(data)
        return df

    except Exception as e:
        print("Fehler bei der Durchführung der Abfragen auf Elasticsearch:", e)


def main(cluster_tweet_data):

    time.sleep(10)
    db = Database()
    micro_clusters = get_micro_clusters_id_and_center(db, cluster_tweet_data)
    macro_micro_dict = macro_clustering(micro_clusters)
    #store_macro_micro_dict_in_database(db, macro_micro_dict)
    #result = get_macro_micro_dict_from_database(db, 'macro_micro_dict')

    print('##################')
    print('Macro-Print:')
    #print(result)
    print(macro_micro_dict)
    print('##################')
