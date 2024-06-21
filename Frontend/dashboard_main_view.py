# Import of function ToDo: Check if all imports are needed
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd
from Frontend.header import get_header
from Microclustering.micro_clustering import get_cluster_tweet_data
from django_plotly_dash import DjangoDash
from Datamanagement.Database import Database


# Setting global variable for last successful generated micro-clustering figure
last_figure = {'data': [], 'layout': go.Layout(title='Number of Tweets per Cluster Over Time',
                                               xaxis=dict(title='Time'),
                                               yaxis=dict(title='Number of Tweets'))}


# Initialize the app
app = DjangoDash('dashboard')

header = get_header()

# App layout
def initialize_dash_app():
    app.layout = html.Div(className='main-body', children=[
        
         # Include the header HTML
        *header,

        # Widget Top Left CSS tl
        html.Div(className='widget', children=[
            html.H3('Micro Cluster'),
            html.Span('Emerging Trends'),
            dcc.Graph(
                    id='live-update-graph',
                    config={'displayModeBar': False},
                    # style={'height': '300px'}
            ),
            dcc.Interval(
                id='interval-component',
                interval=1 * 1000,  # in milliseconds (1 second)
                n_intervals=0
            )
        ]),
        html.Div(className='widget', children=[
                html.H3('Macro Cluster'),
                html.Span('Bag of Words'),

                    # Bag of Words Chart can be embedded here!
                    # Create an extra file and call the function which contains the chart here
                    # So no calculations or similiar in this section. Only calling function
        ]),
        html.Div(className='widget', children=[
                html.H3('Most Recent Posts'),
                html.Span('Post Analysis'),
                    # Most Recent Posts can be embedded here!
                    # Create an extra file and call the function which contains the chart here
                    # So no calculations or similiar in this section. Only calling function
        ]),
        html.Div(className='widget', children=[
                html.H3('Topic Focus'),
                html.Span('Cluster Analysis'),
                    # Topic Focus Widget can be embedded here!
                    # Create an extra file and call the function which contains the chart here
                    # So no calculations or similiar in this section. Only calling function
        ]),
        html.Div(className='widget', children=[
                html.H3('Topic Focus'),
                html.Span('Cluster Analysis'),
                    # Topic Focus Widget can be embedded here!
                    # Create an extra file and call the function which contains the chart here
                    # So no calculations or similiar in this section. Only calling function
        ]),
        html.Div(className='widget', children=[
                html.H3('Topic Focus'),
                html.Span('Cluster Analysis'),
                    # Topic Focus Widget can be embedded here!
                    # Create an extra file and call the function which contains the chart here
                    # So no calculations or similiar in this section. Only calling function
        ]),
        html.Div(className='widget', children=[
                html.H3('Topic Focus'),
                html.Span('Cluster Analysis'),
                    # Topic Focus Widget can be embedded here!
                    # Create an extra file and call the function which contains the chart here
                    # So no calculations or similiar in this section. Only calling function
        ]),
    ])


@app.callback(
        Output('live-update-graph', 'figure'),
        [Input('interval-component', 'n_intervals')]
)

def update_graph_live(n):
    global last_figure

    try:
        # Trying to get cluster data from db
        cluster_tweet_data = get_cluster_tweet_data(db, 'cluster_tweet_data')
        # Ensure 'timestamp' is in datetime format
        cluster_tweet_data['timestamp'] = pd.to_datetime(cluster_tweet_data['timestamp'])

        # Plotting
        traces = []
        for cluster_id in cluster_tweet_data['cluster_id'].unique():
            cluster_data = cluster_tweet_data[cluster_tweet_data['cluster_id'] == cluster_id]
            traces.append(go.Scatter(
                x=cluster_data['timestamp'],
                y=cluster_data['tweet_count'],
                mode='lines+markers',
                name=f'Cluster {cluster_id}'
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

# Callback to toggle dropdown menu
@app.callback(
    Output('dropdown-menu', 'style'),
    [Input('account-toggle', 'n_clicks')]
)
def toggle_dropdown(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# Run App
# TODO: db Instanz aus dem Thread anders anbindbar, sodass keine neue erzeugt werden musss?
db = Database()
initialize_dash_app()
