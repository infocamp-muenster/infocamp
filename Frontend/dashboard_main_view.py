'''
- This files contains the code for the main view of the dashboard.
- The inital Django App named "app" is initialized and all HTML frontend widgets are deployed in this file.
- Major widget as the clustering graphs are updated in this file as well
- Each widget is defined by a Div with CSS class 'widget'
'''

# Import of function ToDo: Check if all imports are needed
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd
from Frontend.header import get_header
from Microclustering.micro_clustering import get_cluster_tweet_data, convert_date
from django_plotly_dash import DjangoDash
from Datamanagement.Database import Database


# Setting global variable for last successful generated micro-clustering figure
last_figure = {'data': [], 'layout': go.Layout(title='Number of Tweets per Cluster Over Time',
                                               xaxis=dict(title='Time'),
                                               yaxis=dict(title='Number of Tweets'))}


# Initialize the app
app = DjangoDash('dashboard')

# HTML Header for Dashboard
header = get_header()

# App layout
def initialize_dash_app():
    app.layout = html.Div(children=[
        
         # Include the header HTML
        *header,

        # Main Div element named 'main-body' contains import style information and all widgets are childs of it
        html.Div(className='main-body', children=[

            html.Div(className='widget', style={'grid-column':'span 9'}, children=[
                    html.H3('KI-Probability'),
                    html.Span('Of Text based Content'),
                        # Widget can be embedded here!
            ]),
            html.Div(className='widget widget-pop-up', children=[
                    html.H3('AI-Generation'),    
            ]),
            # Main Micro Cluster Widget
            html.Div(className='widget', style={'grid-column':'span 9'}, children=[
                html.H3('Micro Cluster'),
                html.Span('Emerging Trends'),
                dcc.Graph(
                        id='live-update-graph',
                        config={'displayModeBar': False},
                ),
                dcc.Interval(
                    id='interval-component',
                    interval=1 * 3000,  # in milliseconds (3 seconds)
                    n_intervals=0
                )
            ]),
            html.Div(className='widget widget-pop-up', id='popup-micro-cluster', children=[
                    html.Span('Micro Cluster Pop Up'),
            ]),

            html.Div(className='widget', style={'grid-column':'span 6'}, children=[
                html.H3('Topic Focus'),
                html.Span('Cluster Analysis'),
                # Widget can be embedded here!
            ]),
            html.Div(className='widget', style={'grid-column':'span 6'}, children=[
                html.H3('Most Recent Posts'),
                html.Span('Post Analysis'),
                    # Widget can be embedded here!
            ]),
            html.Div(className='widget', style={'grid-column':'span 6'}, children=[
                html.H3('Topic Focus'),
                html.Span('Cluster Analysis'),
                    # Widget can be embedded here!
            ]),
            html.Div(className='widget', style={'grid-column':'span 6'}, children=[
                html.H3('Topic Focus'),
                html.Span('Cluster Analysis'),
                    # Widget can be embedded here!
            ]),
        ]),
    ])

# Callback function for updating the realtime micro cluster chart
@app.callback(
        Output('live-update-graph', 'figure'),
        [Input('interval-component', 'n_intervals')]
)

# Function inclduing necessary code for updating micro cluster chart in realtime
def update_graph_live(n):
    global last_figure

    try:
        # Trying to get cluster data from db
        cluster_tweet_data = get_cluster_tweet_data(db, 'cluster_tweet_data')

        # Ensure 'timestamp' is in datetime format
        cluster_tweet_data['timestamp'] = pd.to_datetime(cluster_tweet_data['timestamp'])

        # Predefined line colors
        line_colors_list = ['#07368C', '#707FDD', '#BBC4FD', '#455BE7', '#F1F2FC']

        # Plotting
        traces = []
        for i, cluster_id in enumerate(cluster_tweet_data['cluster_id'].unique()):
            cluster_data = cluster_tweet_data[cluster_tweet_data['cluster_id'] == cluster_id]
            traces.append(go.Scatter(
                x=cluster_data['timestamp'],
                y=cluster_data['tweet_count'],
                mode='lines+markers',
                name=f'Cluster {cluster_id}',
                line=dict(color=line_colors_list[i % len(line_colors_list)]) # Assign color from predefined list
            ))

        layout = go.Layout(
            title='Number of Tweets per Cluster Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Number of Tweets'),
            height=360,  # Höhe des Diagramms in Pixel
        )

        # Update last_figure only if there were no issues while fetching data
        last_figure = {'data': traces, 'layout': layout}

    except Exception as e:
        print(f"An error occurred: {e}")

    return last_figure

# Callback for Micro Cluster Pop Up Widget Information of Datapoints will be shown in widget with id #popup-micro-cluster
@app.callback(
        Output('popup-micro-cluster', 'children'),
        [Input('live-update-graph', 'clickData')]
)

# Function including HTML Output for micro cluster pop up information
def micro_cluster_pop_up(clickData):
    if clickData is None:
        return html.Div(className='widget-pop-up-default',children=[
        html.H4('Click on cluster for detailed information')
    ])
    
    point = clickData['points'][0]
    print(point)
    cluster_number = point['curveNumber']
    cluster_index = point['pointNumber']
    cluster_timestamp = convert_date(point['x'])
    cluster_tweet_count = point['y']

    # Predefined line colors
    line_colors_list = ['#07368C', '#707FDD', '#BBC4FD', '#455BE7', '#F1F2FC']
    cluster_color = line_colors_list[cluster_number]
    
    return html.Div(children=[
        html.H3('Cluster Information'),
        html.Span('Analytics for selected cluster'),
        html.P(children=[
            html.Span(f'Cluster Index:'),
            html.Span(f'{cluster_index}',style={'font-weight':'600'}),
        ]),
        html.P(children=[
            html.Span(f'Cluster Color:'),
            html.Span(f'{cluster_color}',style={'color': cluster_color,'font-weight':'600'}),
        ]),
        html.P(children=[
            html.Span(f'Timestamp:'),
            html.Span(f'{cluster_timestamp}',style={'font-weight':'600'}),
        ]),
        html.P(children=[
            html.Span(f'Tweet Count:'),
            html.Span(f'{cluster_tweet_count}',style={'font-weight':'600'}),
        ]),
    ])


# Run App
# TODO: db Instanz aus dem Thread anders anbindbar, sodass keine neue erzeugt werden musss?
db = Database()
initialize_dash_app()
