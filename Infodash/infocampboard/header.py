# header.py
from dash import html
# Image sources
from PIL import Image
pil_image = Image.open("infocampboard/data/misdoom_logo.png")


def get_header():
    return [ html.Header(className='header', children=[
        html.Img(src=pil_image, alt="Misdoom Logo",style={'filter':'brightness(0) invert(1)'}),
        html.H1('INFOCAMP Dashboard'),
    ]),
    # Header Documentation
    html.Div(className='header-documentation', children=[
        html.Div(className="header-upload-icon"),
        html.Span('Documentation')
    ]),
    # Header Import
    html.Div(className='header-upload', children=[
        html.Span('Upload Data')
    ]),
    # Header Settings
    html.Div(className='header-settings', children=[
        html.A('Logout', href='logout/')
    ]),]