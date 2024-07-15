import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import Infodash.globals as glob
from Datamanagement.Database import Database, get_cluster_tweet_data
import ast
from sklearn.metrics.pairwise import cosine_similarity


def macro_clustering(db, index):
    micro_clusters = get_cluster_tweet_data(db, index)

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
        print("------micro cluster centers np array------")
        print(micro_cluster_centers)

        # Initialize KMeans from Scikit-learn and train it on the micro-cluster centers
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(micro_cluster_centers)

        # Predict the macro-clusters
        macro_clusters = kmeans.predict(micro_cluster_centers)

        # Create a dataframe to store macro-clusters and their belonging micro-clusters
        macro_micro_dict = {}
        macro_cluster_centers = {}
        for macro_cluster, micro_cluster, center in zip(macro_clusters, latest_micro_clusters.iterrows(), micro_cluster_centers):
            if macro_cluster not in macro_micro_dict:
                macro_micro_dict[macro_cluster] = []
                macro_cluster_centers[macro_cluster] = []
            macro_micro_dict[macro_cluster].append(micro_cluster[1]['cluster_id'])
            macro_cluster_centers[macro_cluster].append(center)

        # Compute the center of each macro cluster
        for macro_cluster, centers in macro_cluster_centers.items():
            macro_cluster_centers[macro_cluster] = np.mean(centers, axis=0)
        print("------macro cluster centers------")
        print(macro_cluster_centers)

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(micro_cluster_centers)
        print("------cosine similarity matrix------")
        print(similarity_matrix)

        # Converting dictionary in dataframe
        data = []
        for macro_cluster, micro_clusters_dict in macro_micro_dict.items():
            for micro_cluster in micro_clusters_dict:
                data.append({'macro_cluster': macro_cluster, 'micro_cluster': micro_cluster})

        macro_micro_df = pd.DataFrame(data)

        # Adding tweet_count of every micro-cluster for future visualizing of macro-cluster tweet_count
        micro_cluster_data = pd.DataFrame(micro_clusters)

        tweet_sums = {}

        for micro_cluster in micro_cluster_data['cluster_id'].unique():
            filtered_df = micro_cluster_data[micro_cluster_data['cluster_id'] == micro_cluster]
            tweet_sum = filtered_df['tweet_count'].sum()
            tweet_sums[micro_cluster] = tweet_sum

        macro_micro_df['micro_cluster_tweet_sum'] = macro_micro_df['micro_cluster'].map(tweet_sums)

        return macro_micro_df


def store_macro_micro_dict_in_database(db, macro_micro_dict):
    index_name = 'macro_micro_dict'

    # Dataupload
    try:
        if db.es.indices.exists(index=index_name):
            db.es.indices.delete(index=index_name)
        db.upload_df(index_name, macro_micro_dict)
        glob.macro_df = True
    except Exception as e:
        print(f"An error occurred during upload: {e}")


# TODO: Store macro cluster similarity matrix in database?

def delete_macro_micro_dict_in_database(db):
    index_name = 'macro_micro_dict'
    db.es.indices.delete(index=index_name)


def convert_macro_cluster_visualization(micro_macro_df):
    grouped_df = micro_macro_df.groupby('macro_cluster')['micro_cluster_tweet_sum'].sum().reset_index()
    return grouped_df


def main_macro():
    print("-------------------Started Macro Clustering-------------------")
    db = Database()
    macro_micro_dict = macro_clustering(db, 'cluster_tweet_data')

    store_macro_micro_dict_in_database(db, macro_micro_dict)
