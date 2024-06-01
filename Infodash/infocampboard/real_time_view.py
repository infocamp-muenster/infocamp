# Import of function
from dash import Dash, html, dash_table, dcc
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd
import threading
from infocampboard.tweet_processing import main_loop, get_cluster_tweet_data
from django_plotly_dash import DjangoDash

# Initialize the app
app = DjangoDash('realtimeview')


from PIL import Image
pil_image = Image.open("Infodash/infocampboard/data/misdoom_logo.png")
# App layout

def initialize_dash_app():
    app.layout = html.Div(className='main-body', children=[

        # Header
        html.Header(className='header', children=[
            html.Img(src=pil_image, alt="Misdoom Logo"),
            html.H1('INFOCAMP Dashboard'),
        ]),
        # Header Settings
        html.Div(className='header-settings', children=[
            html.Div(className="header-settings-icon"),
            html.A('Username', href='logout/')
        ]),
        # *header,


            # Main Body
            # Widget Top Left CSS tl
        html.Div(className='widget-tl', children=[
            html.Div(className='widget', children=[
            html.H3('Micro Cluster'),
            html.Span('Emerging Trends'),
            dcc.Graph(
                    id='live-update-graph',
                    config={'displayModeBar': False},
                    # style={'marginTop': '-10px',}
            ),
            dcc.Interval(
                id='interval-component',
                interval=1*2000,  # in milliseconds (10 seconds)
                n_intervals=0
            )
                    # Main Chart Widget can be embedded here!
                    # Create an extra file and call the function which contains the chart here
                    # So no calculations or similiar in this section. Only calling function
            ]),
        ]),
            # Widget Top Left CSS tl
        html.Div(className='widget-tr', children=[
            html.Div(className='widget', children=[
                html.H3('Macro Cluster'),
                html.Span('Bag of Words'),

                    # Bag of Words Chart can be embedded here!
                    # Create an extra file and call the function which contains the chart here
                    # So no calculations or similiar in this section. Only calling function

            ]),
        ]),
            # Widget Bottom Left CSS bl
        html.Div(className='widget-bl', children=[
            html.Div(className='widget', children=[
                html.H3('Most Recent Posts'),

                    # Most Recent Posts can be embedded here!
                    # Create an extra file and call the function which contains the chart here
                    # So no calculations or similiar in this section. Only calling function

            ]),
        ]),
             # Widget Bottom Right CSS br
        html.Div(className='widget-br', children=[
            html.Div(className='widget', children=[
                html.H3('Topic Focus'),

                    # Topic Focus Widget can be embedded here!
                    # Create an extra file and call the function which contains the chart here
                    # So no calculations or similiar in this section. Only calling function

            ]),
        ]),
    ])


    @app.callback(Output('live-update-graph', 'figure'),
                  [Input('interval-component', 'n_intervals')])
    def update_graph_live(n):
        cluster_tweet_data = get_cluster_tweet_data()
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
            xaxis=dict(title='Time'),
            yaxis=dict(title='Number of Tweets'),
            height=400,  # Höhe des Diagramms in Pixel
        )

        return {'data': traces, 'layout': layout}
    return app

def background_process():
    main_loop()
