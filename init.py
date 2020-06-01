import dash
import flask
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.BOOTSTRAP]

server = flask.Flask(__name__)
app = dash.Dash(__name__,
                server=server,
                external_stylesheets=external_stylesheets,
                meta_tags=[{"name": "viewport",
                            "content": "width=device-width, initial-scale=1"}])

app.config.suppress_callback_exceptions = True

app.title = 'CADSEA 2020'

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        
        <script data-ad-client="ca-pub-9632121950956194" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>

        {%favicon%}
        {%css%}
    </head>
    
    <body>
        <div>CADSEA 2020</div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <div>@CADSEA 2020</div>
    </body>
    
</html>
'''

app.scripts.append_script({
    "external_url": "google_advertisement.js"
    })