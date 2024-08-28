from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse

from Datamanagement.Database import Database
from Datamanagement.mapping import map_data_to_dataframe
from .forms import CSVUploadForm
from django.contrib.auth.hashers import make_password
from Microclustering.micro_clustering import export_data
import csv
import json
import os
import io
import time
import pandas as pd

from .models import UploadedData


# LoginPage (Detects if Login or Signup. User database call)
def loginPage(request):
    page = request.GET.get('page', 'login')
    
    if request.user.is_authenticated:
        return redirect('realtime')
    
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
                return redirect('realtime')
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
def upload(request, upload_complete_event):
    data = []
    message = ""
    tweet_count = 0
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            file_name = os.path.splitext(csv_file.name)[0]
            index_name = "data_import"

            timestamp_key = request.POST['timestamp_key']
            username_key = request.POST['username_key']
            user_id_key = request.POST['user_id_key']
            post_id_key = request.POST['post_id_key']
            text_key = request.POST['text_key']

            file_extension = os.path.splitext(csv_file.name)[1].lower()
            if file_extension == '.csv':
                decoded_file = csv_file.read().decode('utf-8').splitlines()
                # Sniffer to detect delimiter
                sample = '\n'.join(decoded_file[:10])  # Use the first 10 lines as a sample
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample)
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = ';'  # Fallback delimiter

                reader = csv.DictReader(decoded_file, delimiter=delimiter)
                for row in reader:
                    data.append(row)
                tweet_count = len(data)  # Anzahl der Tweets zählen
            elif file_extension == '.json':
                data = json.load(csv_file)
                tweet_count = len(data)  # Anzahl der Tweets zählen
            else:
                message = "Unsupported file format. Only JSON and CSV are supported."
                return render(request, 'Frontend/upload.html', {'form': form, 'data': data, 'message': message})

            # Data Mapping to DataFrame
            df = map_data_to_dataframe(
                data,
                timestamp_key,
                username_key,
                user_id_key,
                post_id_key,
                text_key
            )

            # Save Data to ES using DataFrame
            db = Database()
            if db.es.indices.exists(index=index_name):
                db.es.indices.delete(index=index_name)
            db.upload_df(index=index_name, dataframe=df)

            # Warten bis Elasticsearch die korrekte Anzahl von Dokumenten hat
            while True:
                es_count = db.es.count(index=index_name)['count']
                print(f"Current document count in '{index_name}': {es_count}")
                if es_count >= tweet_count:
                    break
                time.sleep(1)  # Warte 1 Sekunde bevor erneut geprüft wird

            # Abrufen einiger Dokumente zur Überprüfung
            results = db.es.search(index=index_name, body={"query": {"match_all": {}}}, size=10)
            for doc in results['hits']['hits']:
                print(doc['_source'])

            message = "Ihr Datensatz wurde erfolgreich hochgeladen!"

            # Set the upload complete event
            upload_complete_event.set()
    else:
        form = CSVUploadForm()

    return render(request, 'Frontend/upload.html', {'form': form, 'data': data, 'message': message})



def dataExport(request):
    # Aufrufen der Funktion export_data() im microclustering
    data = export_data()
    
    # Erstellen der CSV Datei
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False, sep=';')
    csv_buffer.seek(0)

    # Erstellen des HTTPResponse
    response = HttpResponse(csv_buffer, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="data-export.csv"'

    return response
