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
from Microclustering.micro_clustering import convert_date
from Macroclustering.macro_clustering_using_database import convert_macro_cluster_visualization
from django_plotly_dash import DjangoDash
from Datamanagement.Database import Database, get_cluster_tweet_data, get_micro_macro_data
import Infodash.globals as glob

# Define initial empty figures
empty_figure = {
    'data': [],
    'layout': go.Layout(
        title='Loading...',
        xaxis=dict(title=''),
        yaxis=dict(title='')
    )
}

last_chart = go.Figure()

# Initialize the app
app = DjangoDash('dashboard')

# App layout
def initialize_dash_app():
    app.layout = html.Div(children=[

        # Main Div element named 'main-body' contains import style information and all widgets are childs of it
        html.Div(className='main-body', children=[
            # AI Probability Widget / Graph
            html.Div(className='widget', style={'grid-column':'span 9'}, children=[
                html.H3('AI Probability'),
                html.Span('Of Text based Content'),
                dcc.Graph(
                    id='ai-prob-live-update-graph',
                    config={'displayModeBar': False},
                    figure=empty_figure  # Set initial empty figure
                ),
                dcc.Interval(
                    id='ai-prob-interval-component',
                    interval=1 * 3000,  # in milliseconds (3 seconds)
                    n_intervals=0
                )
            ]),
            html.Div(className='widget widget-pop-up', children=[
                    html.H3('AI-Generation'),    
            ]),
            # Main Micro Cluster Widget
            html.Div(className='widget', style={'grid-column':'span 9'}, children=[
                html.H3('Micro Cluster'),
                html.Span('Emerging Trends'),
                dcc.Graph(
                    id='micro-cluster-live-update-graph',
                    config={'displayModeBar': False},
                    figure=empty_figure  # Set initial empty figure
                ),
                dcc.Interval(
                    id='micro-cluster-interval-component',
                    interval=1 * 3000,  # in milliseconds (3 seconds)
                    n_intervals=0
                )
            ]),
            # Pop Up Widget. Gets activated by clicking on data point of micro cluster widget
            # HTML Output of widget is defined below
            html.Div(className='widget widget-pop-up', id='popup-micro-cluster', children=[
                    html.Span('Micro Cluster Pop Up'),
            ]),
            # Main Macro Cluster Widget
            html.Div(className='widget', style={'grid-column': 'span 6'}, children=[
                html.H3('Macro Cluster'),
                html.Span('Bar Chart'),
                dcc.Graph(
                    id='bar-chart',
                    config={'displayModeBar': False},
                ),
                dcc.Interval(
                    id='interval-component',
                    interval=1 * 10000,  # in milliseconds (10 seconds)
                    n_intervals=0
                )
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

# Callback function for updating the realtime ai prob chart
@app.callback(
        Output('ai-prob-live-update-graph', 'figure'),
        [Input('ai-prob-interval-component', 'n_intervals')]
)

# Function inclduing necessary code for updating ai prob chart in realtime
def ai_prob_update_graph_live(n):

    try:
        # Trying to get cluster data from db
        cluster_tweet_data = get_cluster_tweet_data(db, 'cluster_tweet_data')

        # Ensure 'timestamp' is in datetime format
        cluster_tweet_data['timestamp'] = pd.to_datetime(cluster_tweet_data['timestamp'])

        # Predefined line colors
        line_colors_list = ['#07368C', '#707FDD', '#BBC4FD', '#455BE7', '#F1F2FC']

        # Plotting
        ai_prob_traces = []
        for i, cluster_id in enumerate(cluster_tweet_data['cluster_id'].unique()):
            cluster_data = cluster_tweet_data[cluster_tweet_data['cluster_id'] == cluster_id]

            ai_prob_traces.append(go.Scatter(
                x=cluster_data['timestamp'],
                y=cluster_data['tweet_count'],
                mode='lines+markers',
                name=f'Cluster {cluster_id}',
                line=dict(color=line_colors_list[i % len(line_colors_list)]), # Assign color from predefined list
                customdata=list(zip(cluster_data['center'],cluster_data['lower_threshold'],cluster_data['upper_threshold'],cluster_data['std_dev_tweet_count'])),
            ))

        ai_prob_layout = go.Layout(
            title='Number of Tweets >99% AI Probability',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Number of Tweets'),
            height=360,  # Höhe des Diagramms in Pixel
        )

        # Update last_figure only if there were no issues while fetching data
        ai_prop_last_figure = {'data': ai_prob_traces, 'layout': ai_prob_layout}

    except Exception as e:
        print(f"An error occurred: {e}")

    return ai_prop_last_figure

# Callback function for updating the realtime micro cluster chart
@app.callback(
        Output('micro-cluster-live-update-graph', 'figure'),
        [Input('micro-cluster-interval-component', 'n_intervals')]
)

# Function inclduing necessary code for updating micro cluster chart in realtime
def micro_cluster_update_graph_live(n):

    try:
        # Trying to get cluster data from db
        cluster_tweet_data = get_cluster_tweet_data(db, 'cluster_tweet_data')

        # Ensure 'timestamp' is in datetime format
        cluster_tweet_data['timestamp'] = pd.to_datetime(cluster_tweet_data['timestamp'])

        # Predefined line colors
        line_colors_list = ['#07368C', '#707FDD', '#BBC4FD', '#455BE7', '#F1F2FC']

        # Plotting
        micro_cluster_traces = []
        for i, cluster_id in enumerate(cluster_tweet_data['cluster_id'].unique()):
            cluster_data = cluster_tweet_data[cluster_tweet_data['cluster_id'] == cluster_id]

            micro_cluster_traces.append(go.Scatter(
                x=cluster_data['timestamp'],
                y=cluster_data['tweet_count'],
                mode='lines+markers',
                name=f'Cluster {cluster_id}',
                line=dict(color=line_colors_list[i % len(line_colors_list)]), # Assign color from predefined list
                customdata=list(zip(cluster_data['center'],cluster_data['lower_threshold'],cluster_data['upper_threshold'],cluster_data['std_dev_tweet_count'])),
            ))

        micro_cluster_layout = go.Layout(
            title='Number of Tweets per Cluster Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Number of Tweets'),
            height=360,  # Höhe des Diagramms in Pixel
        )

        # Update last_figure only if there were no issues while fetching data
        micro_cluster_last_figure = {'data': micro_cluster_traces, 'layout': micro_cluster_layout}

    except Exception as e:
        print(f"An error occurred: {e}")

    return micro_cluster_last_figure

# Callback for Micro Cluster Pop Up Widget Information of Datapoints will be shown in widget with id #popup-micro-cluster
@app.callback(
        Output('popup-micro-cluster', 'children'),
        [Input('micro-cluster-live-update-graph', 'clickData')]
)

# Function including HTML Output for micro cluster pop up information
def micro_cluster_pop_up(clickData):
    # Default HTML Output of Widget
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

    # Calculation of CSS-Width-vlaues
    if cluster_std_dev != None and cluster_lower_threshold != None and cluster_upper_threshold != None:
        lower_bound = cluster_tweet_count - cluster_std_dev
        upper_bound = cluster_tweet_count + cluster_std_dev
        lower_bound_percentage = 100 * (lower_bound - cluster_lower_threshold) / (cluster_upper_threshold - cluster_lower_threshold)
        upper_bound_percentage = 100 * (upper_bound - cluster_lower_threshold) / (cluster_upper_threshold - cluster_lower_threshold)
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


# Callback for updating macro-cluster chart
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_chart(n):
    global last_chart

    if glob.macro_df:
        try:
            # Create dataframe and bar chart
            df = get_micro_macro_data(db, 'macro_micro_dict')

            grouped_df = convert_macro_cluster_visualization(df)
            last_chart = px.bar(
                grouped_df,
                y='macro_cluster',
                x='micro_cluster_tweet_sum',
                title='Tweet Sum per Macro Cluster',
                labels={'macro_cluster': 'Macro Cluster', 'micro_cluster_tweet_sum': 'Tweet Sum'},
                text='micro_cluster_tweet_sum',
                orientation='h'
            )

            # Update layout for transparency and bar color
            last_chart.update_traces(marker_color='#07368C')
            last_chart.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(
                    tickmode='linear',
                    dtick=1  # Set the tick interval to 1
                )
            )

            return last_chart

        except Exception as e:
            print(f"An error occurred: {e}")

    return last_chart


# Run App
# TODO: db Instanz aus dem Thread anders anbindbar, sodass keine neue erzeugt werden musss?
db = Database()
initialize_dash_app()
