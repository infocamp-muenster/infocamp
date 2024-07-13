from django.urls import path, include

from Infodash.wsgi import upload_complete_event
from . import views
from functools import partial

upload_with_event = partial(views.upload, upload_complete_event=upload_complete_event)

urlpatterns = [
    path('', views.loginPage, name="login"),
    path('logout/', views.logoutPage, name="logout"),

    path('documentation/', views.documentation, name="documentation"),

    path('realtime', views.realTime, name="realtime"),

    path('django_plotly_dash/', include('django_plotly_dash.urls')),

    path('upload/', upload_with_event, name="upload"),
    path('export/', views.dataExport, name="export"),

]