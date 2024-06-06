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


def initialize_dash_app():
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Live Twitter Cluster Analysis"),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000,  # in milliseconds (1 second)
            n_intervals=0
        )
    ])

    @app.callback(Output('live-update-graph', 'figure'),
                  [Input('interval-component', 'n_intervals')])
    def update_graph_live(n):
        cluster_tweet_data = get_cluster_tweet_data(db, 'cluster_tweet_data')

        # Extracting the '_source' part of each dictionary to create a DataFrame
        data = [item['_source'] for item in cluster_tweet_data]
        df = pd.DataFrame(data)

        while df.empty:
            time.sleep(1)
            cluster_tweet_data = get_cluster_tweet_data(db, 'cluster_tweet_data')

            # Extracting the '_source' part of each dictionary to create a DataFrame
            data = [item['_source'] for item in cluster_tweet_data]
            df = pd.DataFrame(data)

        # Ensure 'timestamp' is in datetime format
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            print(f"Error while accessing timestamp: {e}")

        # Plotting
        traces = []
        try:
            for cluster_id in df['cluster_id'].unique():
                cluster_data = df[df['cluster_id'] == cluster_id]
                traces.append(go.Scatter(
                    x=cluster_data['timestamp'],
                    y=cluster_data['tweet_count'],
                    mode='lines+markers',
                    name=f'Cluster {cluster_id}'
                ))
        except Exception as e:
            print(f"Error while accessing cluster_id: {e}")
            print(df)

        layout = go.Layout(
            title='Number of Tweets per Cluster Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Number of Tweets')
        )

        return {'data': traces, 'layout': layout}

    return app

'''def background_process():
    main_loop()'''

if __name__ == '__main__':

    # Adjust to your own user and id_rsa
    ssh_user = 'bwulf'
    ssh_private_key = '/Users/bastianwulf/.ssh/id_rsa_uni'
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