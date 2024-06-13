# header.py
from dash import html
# Image sources
from PIL import Image
pil_image = Image.open("Frontend/images/misdoom_logo.png")

user_icon = Image.open("Frontend/images/user-solid.png")
upload_icon = Image.open("Frontend/images/upload-solid.png")
file_icon = Image.open("Frontend/images/file-solid.png")


def get_header():
    return [ html.Header(className='header', children=[
        html.Img(src=pil_image, alt="Misdoom Logo",style={'filter':'brightness(0) invert(1)'}),
        html.H1('INFOCAMP Dashboard'),
    ]),
    # Header Documentation
    html.Div(className='header-documentation', children=[
        html.Img(src=file_icon, alt="File Icon",style={'width':'15px','margin-right':'10px'}),
        html.Span('Documentation')
    ]),
    # Header Import
    html.Div(className='header-upload', children=[
        html.Img(src=upload_icon, alt="Upload Icon",style={'width':'15px','margin-right':'10px'}),
        html.Span('Upload Data')
    ]),
    # Header Settings
    html.Div(className='header-settings', children=[
        html.Img(src=user_icon, alt="User Icon",style={'width':'15px','margin-right':'10px'}),
        html.A('Logout', href='logout/')
    ]),]