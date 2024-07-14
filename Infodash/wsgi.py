# wsgi.py
"""
WSGI config for Infodash project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application
from Microclustering.thread_micro_clustering import Micro_Clustering_Thread
from Microclustering.ssh_connection import create_ssh_tunnel, stop_ssh_tunnels

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Infodash.settings')

application = get_wsgi_application()

# SSH-Verbindung aufbauen
tunnel1, tunnel2 = create_ssh_tunnel()

# Micro-Clustering-Thread starten
my_thread = Micro_Clustering_Thread()
my_thread.setDaemon(True)
my_thread.start()

# Sicherstellen, dass SSH-Tunnel bei Beendigung des Programms gestoppt werden
def stop_tunnels_on_exit():
    stop_ssh_tunnels(tunnel1, tunnel2)

import atexit
atexit.register(stop_tunnels_on_exit)
