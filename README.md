# INFOCAMP - Board

## Motivation
Social media platforms are becoming the dominant source of information for a significant proportion of the global population. With the deliberate dissemination of false and harmful information, it is crucial that individuals are made aware and that public discourse in online media can continue without malicious intent. We are therefore developing a dashboard to visualize disinformation campaigns in social media posts in real time, utilizing emerging technologies such as word embedding methods like Word2vec and stream clustering. The dashboard will allow users to upload their own social media data and analyse it for anomalies in the activity patterns of related users. Furthermore, it will enable the integration of the probability of AI-generated content to detect disinformation campaigns. Our goal is to aid e.g. computational social scientists to detect and combat disinformation in online media.

## Installation
The dashboard runs via a private server of the University of Muenster. This means that additional authentication must be carried out at the University of Muenster. After that you can run the application locally.
You have to be part of the University of Muenster and have an account at the University in order to use the Dashboard.

### Connecting to Server
1. Create SSH-Key locally
2. Store the SSH-Key in the IT portal of the University of Muenster (https://it-portal.uni-muenster.de/index.php)

### Run the Dashboard locally
1. Clone the repository:
```bash
git clone https://github.com/MattisSipp/infocamp.git
```
2. Log in via your user name and personal access token
3. Change your working directory to the location of the cloned repository
4. In the file Microclustering/ssh_tunnel.py change lines 12 & 13 to your personal information:
```bash
ssh_user = # 'uni-id'
ssh_private_key = # 'path of your SSH-Key'
```
5. install all necessary packages: see the file 'necessary_packages.txt.'
You can use the requirements.txt with pip (for much quicker pip package installation)
```bash
pip install -r requirements.txt
```
6. Run ```python manage.py runserver``` or ```python3 manage.py runserver``` depending on your environment
7. Open link ```http://127.0.0.1:8000/``` in any browser to open the Dashboard
8. Use this data to log in:
   - Username: admin
   - Passwort: infocamp2024
9. Play around and analyze some data!

## Short explanation of different widgets
- AI-Probality Graph: Plots the number of tweets which have likely been created by AI over time. We use an AI-detector by developed by Christian Grimme. The model analyses for each tweet how likely it is that it is AI-generated. We take into account all tweets which have a probability of >99% of being AI-generated. You can click on the peaks of the graph and after that the widget on the right shows recent posts. We aim to detect peaks in AI-generated tweets to see parallels in time and content in this group of tweets.
- Micro Cluster Graph: We aim to detect peaks in specific topics. You can see different topics trending and evolving over time and by clicking on a peak the user can label the cluster to have another look at it later.
- Macro Cluster Graph: Based on the Micro Clusters we develop Macro Clusters, which are the most discussed topics. Via a heat map the differences and similarities between the Macro Clusters are shown.

## Contribution

- [Installation via GitHub](#Installation-via-GitHub)
- [Contribution](#Contribution)

### Installation via GitHub

To install INFOCAMP - Board, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/MattisSipp/infocamp.git
```
2. Switch to your branch:
```bash
git checkout your-branch-name
```

### Contribute to the Dashboard

To contribute, follow these guidelines:

1. Create a new branch for your feature or bug fix:
- For feature branches: `feature/your-feature-name`
- For bugfix branches: `bugfix/your-bugfix-name`

2. Implement your changes and ensure all tests pass.

3. Push your branch to the repository:
```bash
git push origin your-branch-name
```
4. Submit a pull request targeting the `main` branch.

Only admins have permission to push to the `master` branch to ensure stability and reliability of the main branch.

## Info for Developers 
![INFOCAMP - Board](Readme/Micro-Clustering_Sequenzdiagramm.pdf)

## Further exciting things to do with the Dashboard
We were able to create a first running application of the dashboard and have a couple of analyses. If you like our approach and want to enhance the dashboard even further, we listed some possible next steps:
1. Run the dashboard on a public server so that anyone (with proper authentification) can access the dashboard and must not be a part of the University of Muenster
2. Implement user roles and extend the data base so that several people with different data can work with the dashboard simultaneously. 


