'''
- This files contains the code for the main view of the dashboard.
- The inital Django App named "app" is initialized and all HTML frontend widgets are deployed in this file.
- Major widget as the clustering graphs are updated in this file as well
- Each widget is defined by a Div with CSS class 'widget'
'''

from datetime import datetime
import plotly.express as px
from dash import dcc, html, dash_table
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go
import pandas as pd

from Frontend.OpenAI_API import summarize_tweets
from Macroclustering.macro_clustering_using_database import convert_macro_cluster_visualization
from Microclustering.micro_clustering import export_data
from django_plotly_dash import DjangoDash
from Datamanagement.Database import Database, get_cluster_tweet_data, get_micro_macro_data
import Infodash.globals as glob

# Define initial empty figures
empty_figure = {
    'data': [],
    'layout': go.Layout(
        title={
            'text': 'Loading...',
            'font': {
                'family': 'Inter, sans-serif',
                'size': 18,
                'color': '#1F384C'
            }
        },
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=360,
    )
}

ai_prop_last_figure = empty_figure
micro_cluster_last_figure = empty_figure

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
                    id='interval-component',
                    interval=1 * 30000,  # in milliseconds (30 seconds)
                    n_intervals=0
                )
            ]),
            # Pop Up Widget. Gets activated by clicking on data point of ai prob data point
            # HTML Output of widget is defined below
            html.Div(className='widget widget-pop-up', id='popup-ai-prob'),
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
                    id='interval-component',
                    interval=1 * 30000,  # in milliseconds (30 seconds)
                    n_intervals=0
                )
            ]),
            # Pop Up Widget. Gets activated by clicking on data point of micro cluster widget
            # HTML Output of widget is defined below
            html.Div(className='widget widget-pop-up', children=[
                dcc.Tabs(id="popup-tabs", value='tab-1', children=[
                    dcc.Tab(label='Cluster Information', value='tab-1', children=[
                        html.Div(id='popup-micro-cluster')
                    ]),
                    dcc.Tab(label='AI-Summary', value='tab-2', children=[
                        html.Div(id='tab-2-content')
                    ]),
                    dcc.Tab(label='Most Recent Posts', value='tab-3', children=[
                        html.Div(id='tab-3-content')
                    ]),
                ]),
            ]),
            html.Div(className='widget', style={'grid-column': 'span 6'}, children=[
                html.H3('Macro Cluster'),
                html.Span('Bar Chart'),
                dcc.Graph(
                    id='macro-cluster-live-update-graph',
                    config={'displayModeBar': False},
                    figure=empty_figure  # Set initial empty figure
                ),
                dcc.Interval(
                    id='macro-cluster-interval-component',
                    interval=1 * 30000,  # in milliseconds (30 seconds)
                    n_intervals=0
                )
            ]),
            html.Div(className='widget', style={'grid-column': 'span 6'}, children=[
                html.H3('Macro Cluster'),
                html.Span('Heatmap'),
                dcc.Graph(
                    id='macro-cluster-live-update-heatmap',
                    config={'displayModeBar': False},
                    figure=empty_figure  # Set initial empty figure
                ),
                dcc.Interval(
                    id='macro-cluster-interval-component',
                    interval=1 * 10000,  # in milliseconds (10 seconds)
                    n_intervals=0
                )
            ]),
        ]),
    ])

def convert_date(date_str):
    # Parse the input date string to a datetime object
    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')

    # Format the datetime object to the desired output format
    european_format_date_str = dt.strftime('%d.%m.%Y %H:%M')

    return european_format_date_str

# Callback and calculation functions for all widgets
# -- AI PROBABILITY WIDGET --
@app.callback(
Output('ai-prob-live-update-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def ai_prob_update_graph_live(n):
    global ai_prop_last_figure
    try:
        # Trying to get cluster data from db
        cluster_tweet_data = get_cluster_tweet_data(db, 'cluster_tweet_data')
        print(cluster_tweet_data)

        # Ensure 'timestamp' is in datetime format
        cluster_tweet_data['timestamp'] = pd.to_datetime(cluster_tweet_data['timestamp'])
        ai_abs_counter = cluster_tweet_data.groupby('timestamp')['ai_abs'].sum().reset_index()

        # Group by 'timestamp' to get the count of rows per timestamp
        rows_per_timestamp = cluster_tweet_data.groupby('timestamp').size().reset_index(name='rows_per_timestamp')

        # Merge the two dataframes on 'timestamp'
        ai_prob_df = pd.merge(ai_abs_counter, rows_per_timestamp, on='timestamp')



        # Plotting
        ai_prob_traces = go.Scatter(
            x=ai_prob_df['timestamp'],
            y=ai_prob_df['ai_abs'],
            mode='lines+markers',
            name='AI Prob',
            customdata=list(zip(ai_prob_df['rows_per_timestamp'])),
        )

        ai_prob_layout = go.Layout(
            title={
                'text': 'Number of Tweets >99% AI Probability',
                'font': {
                    'family': 'Inter, sans-serif',
                    'size': 18,
                    'color': '#1F384C'
                }
            },
            xaxis=dict(
                title='Time',
                showgrid=True,  # Show grid lines on the x-axis
                gridcolor='lightgray',  # Grid color (adjust as needed)
                gridwidth=1  # Grid line width
            ),
            yaxis=dict(
                title='Number of Tweets',
                range=[0, 20],#TODO: set max y axis value to max value of df
                showgrid=True,  # Show grid lines on the y-axis
                gridcolor='lightgray',  # Grid color (adjust as needed)
                gridwidth=1  # Grid line width
            ),
            height=360,
            font=dict(
                family="Inter, sans-serif",
                size=14,
                color="#1F384C"
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # Erstellung des Figure-Objekts
        ai_prop_last_figure = go.Figure(data=[ai_prob_traces], layout=ai_prob_layout)

    except Exception as e:
        print(f"An error occurred: {e}")

    return ai_prop_last_figure

# -- AI PROBABILITY WIDGET POP UP --
@app.callback(
        Output('popup-ai-prob', 'children'),
        [Input('ai-prob-live-update-graph', 'clickData')]
)
def ai_prob_pop_up(clickData):
    # Default HTML Output of Widget
    if clickData is None:
        return html.Div(className='widget-pop-up-default', children=[
            html.H4('Click on a data point in AI Probabilty widget for more detailed information')
        ])

    point = clickData['points'][0]
    number_of_total_tweets_per_time = point['customdata'][0]
    cluster_timestamp = point['x']
    cluster_tweet_count = point['y']

    # HTML Output of Pop Up Widgets
    return html.Div(children=[
        html.H3('AI Probability Information'),
        html.Span('Analytics for selected data point'),
        html.Div(className="popup-widget-info",children=[
            html.Div(children=[
                html.Span(f'Timestamp:',className="label"),
                html.Span(f'{cluster_timestamp}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Tweet Count:',className="label"),
                html.Span(f'{cluster_tweet_count}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Total Count of Tweets:', className="label"),
                html.Span(f'{number_of_total_tweets_per_time}', className="value"),
            ]),
            html.Div(children=[
                html.Span(f'This chart displays the absolut number of tweets which have an AI probability of more then 99% as per our model. ', className="label"),
            ]),
        ]),
    ])


# -- MICRO CLUSTER WIDGET --
@app.callback(
        Output('micro-cluster-live-update-graph', 'figure'),
        [Input('interval-component', 'n_intervals')]
)
def micro_cluster_update_graph_live(n):
    global micro_cluster_last_figure
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
                customdata=list(zip(cluster_data['lower_threshold'],cluster_data['upper_threshold'],cluster_data['std_dev_tweet_count'])),
            ))

        micro_cluster_layout = go.Layout(
            title={
                'text': 'Number of Tweets per Cluster Over Time',
                'font': {
                    'family': 'Inter, sans-serif',
                    'size': 18,
                    'color': '#1F384C'
                }
            },
            xaxis=dict(title='Time'),
            yaxis=dict(title='Number of Tweets'),
            height=360,
            font=dict(
                family="Inter, sans-serif",
                size=14,
                color="#1F384C"
            )
        )

        # Update last_figure only if there were no issues while fetching data
        micro_cluster_last_figure = {'data': micro_cluster_traces, 'layout': micro_cluster_layout}

    except Exception as e:
        print(f"An error occurred: {e}")

    return micro_cluster_last_figure


# callback for first tab of popup micro cluster
@app.callback(
    Output('popup-micro-cluster', 'children'),
    Input('micro-cluster-live-update-graph', 'clickData')
)
# Function including HTML Output for micro cluster pop up information
def micro_cluster_pop_up(clickData):
    if clickData is None:
        return html.Div(className='widget-pop-up-default', children=[
            html.H4('Click on a data point in Micro Cluster widget for more detailed information')
        ])

    point = clickData['points'][0]
    cluster_number = point['curveNumber']
    cluster_index = point['pointNumber']
    cluster_timestamp = point['x']
    cluster_tweet_count = point['y']

    # Get cluster threshold
    cluster_lower_threshold = point['customdata'][0] if point['customdata'][0] else ""
    cluster_upper_threshold = point['customdata'][1] if point['customdata'][1] else ""

    # Get std_dev_tweet_count
    cluster_std_dev = int(point['customdata'][2]) if point['customdata'][2] else ""

    # HTML Output of Pop Up Widgets
    return html.Div(children=[
        html.H3('Cluster Information'),
        html.Span('Analytics for selected cluster'),
        html.Div(className="popup-widget-info",children=[
            html.Div(children=[
                html.Span(f'Cluster ID:',className="label"),
                html.Span(f'{cluster_number}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Cluster Index:',className="label"),
                html.Span(f'{cluster_index}',className="value"),
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
                html.Span(f'{round(int(cluster_lower_threshold),2)}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Upper Threshold:',className="label"),
                html.Span(f'{round(int(cluster_upper_threshold),2)}',className="value"),
            ]),
            html.Div(children=[
                html.Span(f'Standard Deviation:',className="label"),
                html.Span(f'{cluster_std_dev}', className="value"),
            ]),
            html.Div(children=[
                html.Span(
                    f'This chart displays all micro clusters created by our textclust algorithm based on all tweets per time. ',
                    className="label"),
            ]),
        ]),

        # Display Comments
        html.Div(id='submitted-data', children=[
            html.H3('Comment Section'),
            html.Div(id='comment-list'),
        ], style={'display': 'none'}),

        # Comment Form
        html.H3('Comment Form', style={'margin-bottom': '10px'}),
        html.Div(className="comment-form", children=[
            dcc.Textarea(id='comment', className="value", style={'width': '100%', 'height': 60}),
        ]),
        html.Div(className="comment-form", children=[
            html.Label('Category', className="label"),
            dcc.RadioItems(
                id='category',
                options=[
                    {'label': 'Misinformation', 'value': 'Misinformation'},
                    {'label': 'Desinformation', 'value': 'Desinformation'},
                    {'label': 'Malinformation', 'value': 'Malinformation'},
                ],
                value='Misinformation',  # Default value
                labelStyle={'display': 'block'}
            ),
        ]),
        html.Button('Submit', id='submit-button', n_clicks=0, className="submit-button"),
    ])

# callback for second tab of popup micro cluster
@app.callback(
    Output('tab-2-content', 'children'),
    Input('micro-cluster-live-update-graph', 'clickData')
)

def generate_summary(clickData):
    if clickData is None:
        return html.Div(className='widget-pop-up-default', children=[
            html.H4('Click on a data point in Micro Cluster widget for Summary.')
        ])
    point = clickData['points'][0]
    print('point:')
    print(point)

    # filter only tweets with same micro-cluster
    tweets = export_data()  # Assuming this returns a DataFrame
    point = clickData['points'][0]
    cluster_index = point['pointNumber']
    filt_tweets = tweets[
        (tweets['cluster_id'] == cluster_index)
    ][['text', 'tweet_id']]
    filt_tweets = filt_tweets.drop_duplicates(subset='tweet_id')    # Drop duplicate tweet_ids



    # Convert filt_tweets to a single string
    messages = " ".join(filt_tweets['text'].tolist())

    # Pass the list of dictionaries to the summarize_tweets function
    summary_content = summarize_tweets(messages)
    print('summary_content:')
    print(summary_content)

    return html.Div(summary_content)

@app.callback(
    Output('tab-3-content', 'children'),
    Input('micro-cluster-live-update-graph', 'clickData')
)
def update_recent_posts(clickData):
    if clickData is None:
        return html.Div(className='widget-pop-up-default', children=[
            html.H4('Click on a data point in Micro Cluster widget for Recent Posts.')
        ])

    # Filter tweets based on some condition (example given)
    tweets = export_data()  # Assuming this returns a DataFrame

    #filter only tweets with same micro-cluster and timestamp
    point = clickData['points'][0]

    timestamp = point['x']
    #print('timestamp:')
    #print(timestamp)

    cluster_index = point['pointNumber']
    #print('cluster_index:')
    print(cluster_index)

    #only show column tweet id and text
    # Filter tweets based on timestamp and cluster_index
    filt_tweets = tweets[
        #(tweets['timestamp'] == timestamp) &
        (tweets['cluster_id'] == cluster_index)
        ][['text', 'tweet_id', 'timestamp']]

    # Drop duplicate tweet_ids
    filt_tweets = filt_tweets.drop_duplicates(subset='tweet_id')

    # Sort the filtered DataFrame by 'timestamp' in descending order
    filt_tweets = filt_tweets.sort_values(by='timestamp', ascending=False)

    print('filt_tweets:')
    print(filt_tweets)

    # Convert the DataFrame to a format suitable for DataTable
    posts_content = dash_table.DataTable(
        data=filt_tweets.to_dict('records'),
        columns=[{'name': 'text', 'id': 'text'}, {'name': 'tweet_id', 'id': 'tweet_id'}, {'name': 'timestamp', 'id': 'timestamp'}],
        page_size=10,
        style_cell={
            'textAlign': 'left'}, # Align text to the left
        sort_action='custom',
        sort_mode='single')

    # Wrap the DataTable in an HTML Div for rendering
    return html.Div([posts_content])

# empty list for comments
comments_list = []

# Callback to update displayed comments
@app.callback(
    Output('submitted-data', 'style'),
    Output('comment-list', 'children'),
    Input('submit-button', 'n_clicks'),
    State('comment', 'value'),
    State('category', 'value'),
    State('comment-list', 'children')
)
def update_display(n_clicks, comment, category, current_children):
    if current_children is None or len(current_children) == 0:
        current_children = []
    if n_clicks > 0:
        new_comment = html.Div([
            html.Span(f'Comment:'),
            html.Span(f'{comment}',style={'font-weight:':'600'}),
            html.Span(f'Category:'),
            html.Span(f'{category}', style={'font-weight:': '600'}),
            html.Span(f'Timestamp:'),
            html.Span(f'{datetime.now()}', style={'font-weight:': '600'}),
        ])
        current_children.append(new_comment)
        return {'display': 'block'}, current_children
    else:
        return {'display': 'none'}, current_children

# -- MACRO CLUSTER WIDGET --
@app.callback(
    Output('macro-cluster-live-update-graph', 'figure'),
    [Input('macro-cluster-interval-component', 'n_intervals')]
)
def macro_cluster_update_graph_live(n):

    # var macro_cluster_update_graph_live needs to be none as long as there is no data available
    if not hasattr(macro_cluster_update_graph_live, "macro_cluster_last_figure"):
        macro_cluster_update_graph_live.macro_cluster_last_figure = None

    if glob.macro_df:
        try:
            # Create dataframe and bar chart
            index = 'macro_micro_dict'
            df = get_micro_macro_data(db, index)

            grouped_df = convert_macro_cluster_visualization(df)
            macro_cluster_last_figure = px.bar(
                grouped_df,
                y='macro_cluster',
                x='micro_cluster_tweet_sum',
                title='Tweet Sum per Macro Cluster',
                labels={'macro_cluster': 'Macro Cluster', 'micro_cluster_tweet_sum': 'Tweet Sum'},
                text='micro_cluster_tweet_sum',
                orientation='h'
            )

            macro_cluster_last_figure.update_traces(marker_color='#5A6ACF')
            macro_cluster_last_figure.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(
                    tickmode='linear',
                    dtick=1
                ),
                font=dict(
                    family="Inter, sans-serif",
                    size=14,
                    color="#1F384C"
                )
            )

            return macro_cluster_last_figure

        except Exception as e:
            print(f"An error occurred: {e}")

    # var macro_cluster_update_graph_live needs to be none as long as there is no data available
    if macro_cluster_update_graph_live.macro_cluster_last_figure is None:
        macro_cluster_update_graph_live.macro_cluster_last_figure = empty_figure

    return macro_cluster_update_graph_live.macro_cluster_last_figure

# -- MACRO CLUSTER HEATMAP --
@app.callback(
    Output('macro-cluster-live-update-heatmap', 'figure'),
    [Input('macro-cluster-interval-component', 'n_intervals')]
)
def macro_cluster_update_heatmap_live(n):

    # var macro_cluster_update_heatmap_live needs to be none as long as there is no data available
    if not hasattr(macro_cluster_update_heatmap_live, "macro_cluster_last_heatmap"):
        macro_cluster_update_heatmap_live.macro_cluster_last_heatmap = None

    if glob.macro_similarity_df:
        try:
            # Create dataframe and bar chart
            index = 'macro_similarity_matrix'
            macro_similarity_matrix = get_micro_macro_data(db, index)

            macro_cluster_last_heatmap = px.imshow(
                macro_similarity_matrix,
                # labels=dict(x="Day of Week", y="Time of Day", color="Productivity"),
            )

            macro_cluster_last_heatmap.update_layout(coloraxis = {'colorscale':'Plotly3'})

            return macro_cluster_last_heatmap

        except Exception as e:
            print(f"An error occurred: {e}")

    # var macro_cluster_update_heatmap_live needs to be none as long as there is no data available
    if macro_cluster_update_heatmap_live.macro_cluster_last_heatmap is None:
        macro_cluster_update_heatmap_live.macro_cluster_last_heatmap = empty_figure

    return macro_cluster_update_heatmap_live.macro_cluster_last_heatmap


# Run App
# TODO: db Instanz aus dem Thread anders anbindbar, sodass keine neue erzeugt werden musss?
db = Database()
initialize_dash_app()