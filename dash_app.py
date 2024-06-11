# dash_app.py
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd
import threading
from micro_clustering import get_cluster_tweet_data, main_loop
from Database import Database
import time

# Setting global variable for last successful generated micro-clustering figure
last_figure = {'data': [], 'layout': go.Layout(title='Number of Tweets per Cluster Over Time',
                                               xaxis=dict(title='Time'),
                                               yaxis=dict(title='Number of Tweets'))}


def initialize_dash_app():
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Live Twitter Cluster Analysis"),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds (1 second)
            n_intervals=0
        )
    ])

    @app.callback(Output('live-update-graph', 'figure'),
                  [Input('interval-component', 'n_intervals')])
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
                yaxis=dict(title='Number of Tweets')
            )

            # Update last_figure only if there were no issues while fetching data
            last_figure = {'data': traces, 'layout': layout}

        except Exception as e:
            print(f"An error occurred: {e}")

        return last_figure

    return app


if __name__ == '__main__':
    # Adjust to your own user and id_rsa
    ssh_user = 'jthier' # 'bwulf'
    ssh_private_key = '/Users/janthier/.ssh/id_rsa_uni_ps_server' # '/Users/bastianwulf/.ssh/id_rsa_uni'
    tunnel1, tunnel2 = Database.create_ssh_tunnel(ssh_user, ssh_private_key)
    tunnel1.start()
    tunnel2.start()

    try:
        # Create Database instance
        db = Database()

        # Start the data fetching in a separate thread
        index_name = "tweets-2022-02-17"
        threading.Thread(target=main_loop, args=(db, index_name), daemon=True).start()

        # Create and run the Dash app in a separate thread
        dash_app = initialize_dash_app()
        dash_app.run_server(debug=True, use_reloader=False)

    except KeyboardInterrupt:
        print("Terminating the program...")

    finally:
        tunnel1.stop()
        tunnel2.stop()
