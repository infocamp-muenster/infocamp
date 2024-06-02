from river import stats
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def get_micro_clusters_id_and_center():
    # Datenbankabfrage
    micro_clusters = 
    for micro_cluster in micro_clusters:
        print(micro_cluster["id"])
        print(micro_cluster["center"])


def macro_clustering(micro_clusters):

    # Only apply macro-clustering in case there are any micro-clusters
    if len(micro_clusters) > 0:

        # Create a numpy array from the micro-cluster centers
        micro_cluster_centers = np.array(micro_clusters["center"])

        # Initialize KMeans from Scikit-learn and train it on the micro-cluster centers
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(micro_cluster_centers)

        # Predict the macro-clusters
        macro_clusters = kmeans.predict(micro_cluster_centers)
        print(f"Macro clusters: {macro_clusters}")

        # Create a dictionary to store macro-clusters and their belonging micro-clusters
        macro_micro_dict = {}
        print(f"zipped: {zip(macro_clusters, micro_clusters)}")
        for macro_cluster, micro_cluster in zip(macro_clusters, micro_clusters):
            if macro_cluster not in macro_micro_dict:
                macro_micro_dict[macro_cluster] = []
            macro_micro_dict[macro_cluster].append(micro_cluster)

        print(f"Macro clusters dictionary: {macro_micro_dict}")

        # Visualize the macro clusters
        plt.scatter(micro_clusters[:, 0], micro_clusters[:, 1], c=macro_clusters)
        plt.show()

        return macro_micro_dict


def store_macro_micro_dict_in_database(macro_micro_dict):
    # Datenbankanfrage
    return


def main():
    micro_clusters = get_micro_clusters_id_and_center()
    macro_micro_dict = macro_clustering(micro_clusters)
    store_macro_micro_dict_in_database(macro_micro_dict)