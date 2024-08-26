'''
- This files contains the code for the main view of the dashboard.
- The inital Django App named "app" is initialized and all HTML frontend widgets are deployed in this file.
- Major widget as the clustering graphs are updated in this file as well
- Each widget is defined by a Div with CSS class 'widget'
'''

# Import of function ToDo: Check if all imports are needed
from dash import dcc, html, dash_table
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd

from Frontend.OpenAI_API import summarize_tweets
from Frontend.header import get_header
from Microclustering.micro_clustering import get_cluster_tweet_data, convert_date, export_data
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
            #Pop up widget with three different tabs Cluster Inf., KI Summ. and recent posts of the micro cluster point
            html.Div(className='widget widget-pop-up', id='popup-micro-cluster', children=[
                dcc.Tabs(id="popup-tabs", value='tab-1', children=[
                    dcc.Tab(label='Cluster Information', value='tab-1', children=[
                        html.Div(id='tab-1-content')
                    ]),
                    dcc.Tab(label='KI-Summary', value='tab-2', children=[
                        html.Button('Generate Summary', id='generate-summary-button', n_clicks=0),
                        html.Div(id='tab-2-content')
                    ]),
                    dcc.Tab(label='Most Recent Posts', value='tab-3', children=[
                        html.Div(id='tab-3-content')
                    ]),
                ]),
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
                line=dict(color=line_colors_list[i % len(line_colors_list)]), # Assign color from predefined list
                customdata=list(zip(cluster_data['center'],cluster_data['lower_threshold'],cluster_data['upper_threshold'],cluster_data['std_dev_tweet_count'])),
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

# Unified callback for popup micro cluster with tabs
@app.callback(
    [Output('tab-1-content', 'children'),
     Output('tab-3-content', 'children')],
    [Input('popup-tabs', 'value'),
     Input('live-update-graph', 'clickData')]
)
def update_popup_content(selected_tab, clickData):
    cluster_content = 'Click on cluster for detailed information'
    posts_content = 'Click on a cluster to view the most recent posts.'

    if clickData:
        # Update content for Cluster Information (tab-1)
        cluster_content = micro_cluster_pop_up(clickData)

        # Example logic for posts content
        posts_content = html.Div('Example of most recent posts')

    return cluster_content, posts_content


@app.callback(
    Output('tab-2-content', 'children'),
    [Input('generate-summary-button', 'n_clicks')],
    [Input('live-update-graph', 'clickData')]
)
def generate_summary(n_clicks, clickData):
    if n_clicks > 0 and clickData:
        point = clickData['points'][0]
        cluster_key_words_string = ", ".join(point['customdata'][0].keys()) if point['customdata'][0] else ""
        summary_content = summarize_tweets(cluster_key_words_string)
        return summary_content

    # Default message before button is clicked
    return ''


# Function including HTML Output for micro cluster pop up information
def micro_cluster_pop_up(clickData):
    if clickData is None:
        return html.Div(className='widget-pop-up-default', children=[
            html.H4('Click on cluster for detailed information')
        ])

    point = clickData['points'][0]
    cluster_number = point['curveNumber']
    cluster_index = point['pointNumber']
    cluster_timestamp = convert_date(point['x'])
    cluster_tweet_count = point['y']

    # Get cluster keywords
    cluster_key_words_string = ", ".join(point['customdata'][0].keys()) if point['customdata'][0] else ""

    # Get cluster threshold
    cluster_lower_threshold = point['customdata'][1] if point['customdata'][1] else ""
    cluster_upper_threshold = point['customdata'][2] if point['customdata'][2] else ""

    # Get std_dev_tweet_count
    cluster_std_dev = int(point['customdata'][3]) if point['customdata'][3] else ""

    # Calculation of CSS-Width-values
    if cluster_std_dev != None and cluster_lower_threshold != None and cluster_upper_threshold != None:
        lower_bound = cluster_tweet_count - cluster_std_dev
        upper_bound = cluster_tweet_count + cluster_std_dev

        lower_bound_percentage = 100 * (lower_bound - cluster_lower_threshold) / (
                    cluster_upper_threshold - cluster_lower_threshold)
        upper_bound_percentage = 100 * (upper_bound - cluster_lower_threshold) / (
                    cluster_upper_threshold - cluster_lower_threshold)
        width_percentage = upper_bound_percentage - lower_bound_percentage
    else:
        width_percentage = 100
        lower_bound_percentage = 0

    # Predefined line colors
    line_colors_list = ['#07368C', '#707FDD', '#BBC4FD', '#455BE7', '#F1F2FC']
    cluster_color = line_colors_list[cluster_number]

    # HTML Output of Pop Up Widgets
    return html.Div(children=[
        html.H3('Cluster Information'),
        html.Span('Analytics for selected cluster'),
        html.Div(className="popup-widget-info",children=[
            html.Div(children=[
                html.Span(f'Cluster Index:',className="label"),
                html.Span(f'{cluster_index}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Cluster Color:',className="label"),
                html.Span(f'{cluster_color}',style={'color': cluster_color},className="value"),
            ]),
            html.Div(className="keywords",children=[ # Extra CSS class for larger width handling
                html.Span(f'Cluster Keywords:',className="label"),
                html.Span(f'{cluster_key_words_string}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Timestamp:',className="label"),
                html.Span(f'{cluster_timestamp}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Tweet Count:',className="label"),
                html.Span(f'{cluster_tweet_count}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Lower Threshold:',className="label"),
                html.Span(f'{round(cluster_lower_threshold,2)}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Upper Threshold:',className="label"),
                html.Span(f'{round(cluster_upper_threshold,2)}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Standard Deviation:',className="label"),
                html.Span(f'{round(cluster_std_dev,2)}', className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Position in Cluster:', className="label"),
                html.Div(className="threshold-bar", children=[
                        html.Div(className="threshold-bar-inner",style={'width': f'{round(width_percentage,0)}%','left': f'{round(lower_bound_percentage)}%'}),
                ])
            ]),
        ]),
    ])

if __name__ == '__main__':
    app.run_server(debug=True)

# Run App
# TODO: db Instanz aus dem Thread anders anbindbar, sodass keine neue erzeugt werden musss?
db = Database()
initialize_dash_app()
