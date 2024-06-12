"""
WSGI config for Infodash project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
from infocampboard.thread_micro_clustering import Micro_Clustering_Thread


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Infodash.settings')

application = get_wsgi_application()

my_thread = Micro_Clustering_Thread()
my_thread.setDaemon(True)
my_thread.start()