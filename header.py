# sidebar.py
from dash import html

def get_header():
    return[ html.Link(
        rel='stylesheet',
        href='static/style.css'
    ),
    # Header
    html.Header(className='header', children=[
        html.H1('INFOCAMP Dashboard'),
        html.Span('Real-time Visualization of Desinformation Campaigns')
    ]),
    # Header Settings
    html.Div(className='header-settings', children=[
        html.Div(className="header-settings-icon"),
        html.Span('Username')
    ]),]
    