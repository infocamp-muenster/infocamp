import os
from django.core.wsgi import get_wsgi_application
from threading import Event
from Microclustering.ssh_tunnel import SSHTunnelThread
from Microclustering.thread_micro_clustering import MicroClusteringThread

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Infodash.settings')

application = get_wsgi_application()

# Create event objects
ssh_ready_event = Event()
upload_complete_event = Event()

# Start the SSH tunnel in a separate thread and pass the ssh_ready_event
ssh_tunnel_thread = SSHTunnelThread(ssh_ready_event)
ssh_tunnel_thread.setDaemon(True)
ssh_tunnel_thread.start()

# Start the micro clustering thread but it will wait until the upload_complete_event is set
micro_clustering_thread = MicroClusteringThread(ssh_ready_event, upload_complete_event)
micro_clustering_thread.setDaemon(True)
micro_clustering_thread.start()