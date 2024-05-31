# Import of function
from dash import Dash, html, dash_table, dcc
import plotly.express as px
from django_plotly_dash import DjangoDash
import pandas as pd

from PIL import Image
pil_image = Image.open("../infocampboard/images/misdoom_logo.png")

df = pd.read_excel('../infocampboard/data/Excel_tweets.xlsx', sheet_name='Sheet1')

# Initialize the app
app = DjangoDash('Content-Analysis')


# App layout
app.layout = html.Div(className='main-body', children=[

    
    # Header
    html.Header(className='header', children=[
        html.H1('INFOCAMP Dashboard'),
        html.Span('Content Analysis of Desinformation Campaigns')
    ]),
    # Header Settings
    html.Div(className='header-settings', children=[
        html.Div(className="header-settings-icon"),
        html.A('Username', href='#')
    ]),
    html.Aside(className='sidebar', children=[
        html.Img(src=pil_image, alt="Misdoom Logo"),
        html.Ul([
            html.Li(html.A('Real-time View', href='realtime', className='active-tab')),
            html.Li(html.A('Content Analysis', href='contentanalysis')),
            html.Li(html.A('AI Content', href='#')),
            html.Li(html.A('Account Analysis', href='#')),
        ]),
        html.Div(className='import-section', children=[
            html.A('Import Data', href='#')
        ])
    ]),

    # Main Body
    # Widget Top Left CSS tl
    html.Div(className='widget-tl', children=[
        html.Div(className='widget', children=[
            html.H3('Content Analysis'),
            
        ]),
    ]),
    # Widget Top Left CSS tl
    html.Div(className='widget-tr', children=[
        html.Div(className='widget', children=[
            html.H3('Cluster Analysis'),
            
        ]),
    ]),
])
