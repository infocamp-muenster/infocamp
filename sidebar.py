# sidebar.py
from dash import html

# Image sources
from data_tmp.data_imp import df
from PIL import Image
pil_image = Image.open("images/misdoom_logo.png")

def get_sidebar():
    return html.Aside(className='sidebar', children=[
        html.Img(src=pil_image, alt="Misdoom Logo"),
        html.Ul([
            html.Li(html.A('Real-time View', href='real-time-view.py')),
            html.Li(html.A('Content Analysis', href='content-analysis.py')),
            html.Li(html.A('AI Content', href='#')),
            html.Li(html.A('Account Analysis', href='#')),
        ])
    ])