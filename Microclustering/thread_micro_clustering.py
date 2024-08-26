import threading
from Microclustering.micro_clustering import main_loop
from Datamanagement.Database import Database

class MicroClusteringThread(threading.Thread):
    def __init__(self, ssh_event, upload_event):
        super().__init__()
        self.ssh_event = ssh_event
        self.upload_event = upload_event

    def run(self):

        micro_algo = "Textclust" # "Clustream"
 
        try:
            # Wait for the SSH tunnel to be established
            self.ssh_event.wait()

            # Wait for the upload to be complete
            self.upload_event.wait()

            # Create Database instance
            db = Database()

            # Start the data fetching and clustering
            index_name = "data_import"
            main_loop(db, index_name, micro_algo)

        except KeyboardInterrupt:
            print("Terminating the program...")