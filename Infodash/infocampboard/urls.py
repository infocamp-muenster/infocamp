from django.urls import path, include
from . import views

urlpatterns = [
    path('login/', views.loginPage, name="login"),
    path('logout/', views.logoutPage, name="logout"),
    
    path('', views.home, name="Home"),
    
    path('realtime', views.realTime, name="Realtime"),
    path('contentanalysis', views.contentAnalysis, name="Contentanalysis"),
    
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
   
]
