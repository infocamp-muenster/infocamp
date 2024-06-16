from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from .forms import CSVUploadForm
import csv

# Create your views here.

def loginPage(request):
    page = 'login'
    if request.user.is_authenticated:
        return redirect('Realtime')
    
    if request.method == 'POST':
        username = request.POST.get('username').lower()
        password = request.POST.get('password')
        
        try:
            user = User.objects.get(username=username)
        except:
            messages.error(request, 'User does not exist')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('Realtime')
        else:
            messages.error(request, 'Username OR password does not exist')
            
    context = {'page': page}
    return render(request, 'Frontend/login.html', context)

def logoutPage(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def realTime(request):
    return render(request, 'Frontend/dashboard.html')

def documentation(request):
    return render(request, 'Frontend/docu.html')

# Functions returns df uploaded as CSV
def upload(request):
    data = []
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            decoded_file = csv_file.read().decode('utf-8').splitlines()
            reader = csv.DictReader(decoded_file, delimiter=';')
            for row in reader:
                data.append(row)

            # TODO: Datenbank connection und das df dort abspeichern    
            return redirect('Realtime')
    else:
        form = CSVUploadForm()
    return render(request, 'Frontend/upload.html', {'form': form, 'data': data})