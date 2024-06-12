import threading
from Database import Database
from infocampboard.micro_clustering import main_loop
 
class Micro_Clustering_Thread(threading.Thread):
    def run(self):
        ssh_user = 'jthier' # 'bwulf'
        ssh_private_key = '/Users/janthier/.ssh/id_rsa_uni_ps_server' # '/Users/bastianwulf/.ssh/id_rsa_uni'
        tunnel1, tunnel2 = Database.create_ssh_tunnel(ssh_user, ssh_private_key)
        tunnel1.start()
        tunnel2.start()
 
        try:
            # Create Database instance
            db = Database()
 
            # Start the data fetching in a separate thread
            index_name = "tweets-2022-02-17"
            main_loop(db, index_name)
 
        except KeyboardInterrupt:
            print("Terminating the program...")
 
        finally:
            tunnel1.stop()
            tunnel2.stop()