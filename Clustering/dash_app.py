# dash_app.py
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd
import threading
from tweet_processing import main_loop, get_cluster_tweet_data
from PIL import Image

def initialize_dash_app():
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Live Twitter Cluster Analysis"),
        dcc.Graph(id='live-update-graph'),
        html.Img(src=Image.open('wordcloud.png'), style={'width': '100%'}),
        dcc.Interval(
            id='interval-component',
            interval=1*2000,  # in milliseconds (10 seconds)
            n_intervals=0
        )
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
            title='Number of Tweets per Cluster Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Number of Tweets')
        )

        return {'data': traces, 'layout': layout}

    return app

def background_process():
    main_loop()

if __name__ == '__main__':
    bg_thread = threading.Thread(target=background_process)
    bg_thread.daemon = True
    bg_thread.start()

    app = initialize_dash_app()
    app.run_server(debug=True)
