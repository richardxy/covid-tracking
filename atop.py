import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# layout
import page_by_region
import page_search
import page_model
import homepage


from init import app, server


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/by_region':
        return page_by_region.App()
    elif pathname == "/search":
        return page_search.App()
    elif pathname == "/model":
        return page_model.load_layout()
    else:
        return homepage.load_layout()


if __name__ == '__main__':
    app.run_server(debug=False, port=5010)
