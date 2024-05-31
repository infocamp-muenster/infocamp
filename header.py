# sidebar.py
from dash import html
# Image sources
from PIL import Image
pil_image = Image.open("images/misdoom_logo.png")

def get_header():
    return[ html.Link(
        rel='stylesheet',
        href='static/style.css'
    ),
    # Header
    html.Header(className='header', children=[
        html.Img(src=pil_image, alt="Misdoom Logo"),
        html.H1('INFOCAMP Dashboard'),
    ]),
    # Header Settings
    html.Div(className='header-settings', children=[
        html.Div(className="header-settings-icon"),
        html.A('Username', href='#')
    ]),]
    