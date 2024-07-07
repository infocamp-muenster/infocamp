from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm

from Datamanagement.Database import Database
from Datamanagement.mapping import map_data_to_json
from .forms import CSVUploadForm
from django.contrib.auth.hashers import make_password
import csv
import json
import os

from .models import UploadedData


# LoginPage (Detects if Login or Signup. User database call)
def loginPage(request):
    page = request.GET.get('page', 'login')
    
    if request.user.is_authenticated:
        return redirect('Realtime')
    
    if request.method == 'POST':
        if page == 'login':
            username = request.POST.get('username').lower()
            password = request.POST.get('password')
            
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                messages.error(request, 'Username does not exist!')
                return render(request, 'Frontend/login.html', {'page': page})
            
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                login(request, user)
                return redirect('Realtime')
            else:
                messages.error(request, 'Password does not exist!')
        else:
            username = request.POST.get('username').lower()
            password = request.POST.get('password')
            email = request.POST.get('email')
            
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists')
            elif User.objects.filter(email=email).exists():
                messages.error(request, 'Email already registered')
            else:
                user = User.objects.create(
                    username=username,
                    password=make_password(password),
                    email=email
                )
                messages.success(request, 'User registered successfully!')
                return redirect('login')

    context = {'page': page}
    return render(request, 'Frontend/login.html', context)

#Logout Function
def logoutPage(request):
    logout(request)
    return redirect('login')

#Checks Login´
@login_required(login_url='login')
def realTime(request):
    return render(request, 'Frontend/dashboard.html')

#Documentation Function
def documentation(request):
    return render(request, 'Frontend/docu.html')

# Functions returns df uploaded as CSV
@login_required(login_url='login')
@login_required(login_url='login')
def upload(request):
    data = []
    message = ""
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            file_name = os.path.splitext(csv_file.name)[0]
            username = request.user.username
            index_name = f"{username}_{file_name}"

            timestamp_key = request.POST['timestamp_key']
            username_key = request.POST['username_key']
            user_id_key = request.POST['user_id_key']
            post_id_key = request.POST['post_id_key']
            text_key = request.POST['text_key']

            file_extension = os.path.splitext(csv_file.name)[1].lower()
            if file_extension == '.csv':
                decoded_file = csv_file.read().decode('utf-8').splitlines()
                reader = csv.DictReader(decoded_file, delimiter=';')
                for row in reader:
                    data.append(row)
            elif file_extension == '.json':
                data = json.load(csv_file)
            else:
                message = "Unsupported file format. Only JSON and CSV are supported."
                return render(request, 'Frontend/upload.html', {'form': form, 'data': data, 'message': message})

            # Data Mapping
            mapped_data = map_data_to_json(
                data,
                timestamp_key,
                username_key,
                user_id_key,
                post_id_key,
                text_key
            )

            # Save Data to ES
            db = Database()
            db.upload(index=index_name, data=json.loads(mapped_data))

            # Speichern der Metadaten in der Django-Datenbank
            UploadedData.objects.create(
                user=request.user,
                file_name=file_name,
                index_name=index_name
            )

            message = "Ihr Datensatz wurde erfolgreich hochgeladen!"
    else:
        form = CSVUploadForm()

    return render(request, 'Frontend/upload.html', {'form': form, 'data': data, 'message': message})
