# header.py
from dash import html
# Image sources
from PIL import Image
pil_image = Image.open("Frontend/images/misdoom_logo.png")

user_icon = Image.open("Frontend/images/user-solid.png")
upload_icon = Image.open("Frontend/images/upload-solid.png")
export_icon = Image.open("Frontend/images/export-solid.png")
file_icon = Image.open("Frontend/images/file-solid.png")


def get_header():
    return [ 
        html.Div(className='outer-header-sticky', children=[ #Sticky Header needs Div around itself
            html.Header(className='header', children=[
                html.A(href='/', children=[
                    html.Img(src=pil_image, alt="Misdoom Logo",style={'filter':'brightness(0) invert(1)'}),
                    html.H1('INFOCAMP Dashboard'),
                ]),
                html.Div(className='header-right', children=[
                    # Header Documentation
                    html.Div(children=[
                        html.Img(src=file_icon, alt="File Icon",style={'width':'15px',}),
                        html.A('Documentation', href='documentation/')
                    ]),
                    # Header Import
                    html.Div(children=[
                        html.Img(src=upload_icon, alt="Upload Icon",style={'width':'15px',}),
                        html.A('Upload Data', href='upload/')
                    ]),
                    # Header Export
                    html.Div(children=[
                        html.Img(src=export_icon, alt="Export Icon",style={'width':'15px',}),
                        html.A('Export Data', href='export/')
                    ]),
                    # Header Settings
                    html.Div(children=[
                        html.Img(src=user_icon, alt="User Icon",style={'width':'15px',}),
                        html.A('Logout', href='logout/')
                    ]),
                ])
            ]), 
        ]),
    ]