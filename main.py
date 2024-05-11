# Import of function
from dash import Dash, html, dash_table, dcc
import plotly.express as px

# Initialize the app
app = Dash(__name__)

from data_tmp.data_imp import df
from PIL import Image

# Image sources
pil_image = Image.open("images/misdoom_logo.png")

# App layout
app.layout = html.Div(className='dashboard_main', children=[
    html.Link(
        rel='stylesheet',
        href='static/style.css'
    ),
    # Sidebar
    html.Div(className='sidebar', children=[
        html.Img(src=pil_image, alt="Misdoom Logo"),
        html.Hr(className="divider"),
        html.Ul([
            html.Li(html.A('Dashboard', href='#')),
            html.Li(html.A('Desinformationen', href='#')),
            html.Li(html.A('Managing', href='#')),
        ])
    ]),
    # Main Body
    html.Div(className='content', children=[
        # Header
        html.Div(className='header_title', children=[
            html.H1('INFOCAMP Dashboard'),
            html.Span('Real-time Visualization of Desinformation Campaigns')
        ]),
        html.Hr(className="divider"),
        html.Div(className='header', children=[
            html.Ul([
                html.Li(children=[
                    html.Span(className='title', children='Aktive Anzahl von Desinformationskampagnen in DE:'),
                    html.Br(),
                    html.Span(className='value', children='530'),
                ]),
                html.Li(children=[
                    html.Span(className='title', children='Beiträge von heute aus Desinformationskampagnen:'),
                    html.Br(),
                    html.Span(className='value', children='530'),
                ]),
                html.Li(children=[
                    html.Span(className='title', children='Anteil aller Beiträge aus Desinformationskampagnen:'),
                    html.Br(),
                    html.Span(className='value', children='530'),
                ]),
            ])
        ]),
        # Plot section
        html.Div(className='plot', children=[
            # Widget 1: DataTable
            html.Div(className='basic_widget', style={'float': 'left',} ,children=[
                html.H3('Aktuelle Beiträge'),
                dash_table.DataTable(
                    columns=[
                        {'name': 'User', 'id': 'username', 'type': 'text'},
                        {'name': 'Date', 'id': 'inserted_at', 'type': 'datetime'},
                        {'name': 'Time', 'id': 'time', 'type': 'datetime'},
                        {'name': 'Content', 'id': 'text', 'type': 'text'},
                        {'name': 'Topic', 'id': 'topic', 'type': 'text'},
                        {'name': 'Likes', 'id': 'likes', 'type': 'numeric'}
                    ],
                    data=df.to_dict('records'),
                    page_size=10,
                    filter_action='native',
                    style_cell={'font-family': 'Inter, sans-serif','text-align': 'left','fontWeight': '300'},
                    style_header={'font-family': 'Inter, sans-serif','fontWeight': '300','background-color': '#5A6ACF','color:':'#ffffff','border-radius':'20px'}
                ),
            ]),
            # Widget 2: Histogram
            html.Div(className='basic_widget', style={'float': 'right'}, children=[
                html.H3('Histogram Beiträge'),
                dcc.Graph(
                    figure=px.histogram(df, x='topic', y='likes', histfunc='sum'),
                    config={'displayModeBar': False}  # Disable Tool Section
                )
            ])
        ])
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
