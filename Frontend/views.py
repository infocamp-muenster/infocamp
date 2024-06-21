from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from .forms import CSVUploadForm
from django.contrib.auth.hashers import make_password
import csv

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