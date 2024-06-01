from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.loginPage, name="login"),
    path('logout/', views.logoutPage, name="logout"),
    
    path('home/', views.home, name="Home"),
    
    path('realtime', views.realTime, name="Realtime"),
    
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
   
]
