# Import of function
from dash import Dash, html, dash_table, dcc
import plotly.express as px

# Initialize the app
app = Dash(__name__)

# import header from header.py
from header import get_header
header = get_header()

# import sidebar / nav from sidebar.py
from sidebar import get_sidebar
sidebar = get_sidebar(),

# App layout
app.layout = html.Div(className='main-body', children=[

    *header,
    *sidebar,

    # Main Body
    # Widget Top Left CSS tl
    html.Div(className='widget-tl', children=[
        html.Div(className='widget', children=[
            html.H3('Emerging Trends'),
            
        ]),
    ]),
    # Widget Top Left CSS tl
    html.Div(className='widget-tr', children=[
        html.Div(className='widget', children=[
            html.H3('Bag of Words'),
            
        ]),
    ]),
    # Widget Bottom Left CSS bl
    html.Div(className='widget-bl', children=[
        html.Div(className='widget', children=[
            html.H3('Most Recent Posts'),
            
        ]),
    ]),
     # Widget Bottom Right CSS br
    html.Div(className='widget-br', children=[
        html.Div(className='widget', children=[
            html.H3('Topic Focus'),
            
        ]),
    ]),
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
