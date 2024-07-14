import threading
from Datamanagement.Database import Database
from Microclustering.micro_clustering import main_loop

class Micro_Clustering_Thread(threading.Thread):
    def run(self):
        try:
            # Create Database instance
            db = Database()

            # Start the data fetching in a separate thread
            index_name = "tweets-2022-02-17"
            main_loop(db, index_name)

        except KeyboardInterrupt:
            print("Terminating the program...")
