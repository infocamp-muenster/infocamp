from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
]
