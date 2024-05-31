# sidebar.py
from dash import html

# Image sources
from PIL import Image
pil_image = Image.open("images/misdoom_logo.png")

def get_sidebar():
    return html.Aside(className='sidebar', children=[
        html.Img(src=pil_image, alt="Misdoom Logo"),
        html.Ul([
            html.Li(html.A('Real-time View', href='real-time-view.py', className='active-tab')),
            html.Li(html.A('Content Analysis', href='content-analysis.py')),
            html.Li(html.A('AI Content', href='#')),
            html.Li(html.A('Account Analysis', href='#')),
        ]),
        html.Div(className='import-section', children=[
            html.A('Import Data', href='#')
        ])
    ])