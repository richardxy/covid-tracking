import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import os
import utilities as utl
import pickle
from dash.dependencies import Output, Input
from navbar import Navbar
from init import app

nav = Navbar()

if os.path.exists('data/simulation_results.pickle'):
    with open('data/simulation_results.pickle', 'rb') as file:
        sim_data = pickle.load(file)
else:
    sim_data = None

def load_body():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Alert(html.H6('Social Connection'))], md=4
                    ),
                    dbc.Col(
                        [dbc.Alert(html.H6('Recovery Time'))], md=4
                    ),
                    dbc.Col(
                        [dbc.Alert(html.H6('Transmission Time'))], md=4
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(dcc.Dropdown(
                                id='Social Connection',
                                options=[{'label': '5', 'value': 5}, {'label': '10', 'value': 10}, {'label': '50', 'value': 50}],
                                value=5,
                                multi=False

                            ))
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(dcc.Dropdown(
                                id='Recovery Time',
                                options=[{'label': '1', 'value': 1}, {'label': '2', 'value': 2}, {'label': '4', 'value': 4}],
                                value=4,
                                multi=False

                            ))
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(dcc.Dropdown(
                                id='Transmission Time',
                                options=[{'label': '0.1', 'value': 0.1}, {'label': '0.3', 'value': 0.3}, {'label': '0.7', 'value': 0.7}],
                                value=0.1,
                                multi=False

                            ))
                        ],
                        md=4,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id='spread_graph',
                                     children=[],
                                     ))
                ]
            )

        ],
        className="mt-4",
    )


def load_layout():
    layout = html.Div([
        nav,
        load_body(),
    ])
    return layout

@app.callback(
    Output(component_id='spread_graph', component_property='children'),
    [Input(component_id='Social Connection', component_property='value'),
     Input('Recovery Time', 'value'),
     Input('Transmission Time', 'value'), ])
def update_graph(conn_num, rec_time, trans_time):
    print('conn_num:{}, rec_time:{}, trans_time:{}'.format(conn_num, rec_time, trans_time))
    if sim_data is not None:
        res = next(item for item in sim_data if item["conn_num"] == conn_num and item['rec_time'] == rec_time and item['trans_time'] == trans_time)
        if res is None:
            print('canot find res')
        lines = [{'x': res['t'], 'y':res['I'], 'name': 'Infected', 'mode':'lines+markers'},
                 {'x': res['t'], 'y':res['S'], 'name':'Susceptible', 'mode':'lines+markers'},
                 # {'x': res['t'], 'y':res['E'], 'name':'Exposed', 'mode':'lines+markers'},
                 {'x': res['t'], 'y':res['R'], 'name':'Recovered', 'mode':'lines+markers'}, ]
        graph = dcc.Graph(
            figure={
                'data': lines,  # data,
                'layout': go.Layout(
                    title='Social-connection:{}, Trans-time:{}, Recovery-time:{}'.format(
                        conn_num, trans_time, rec_time),
                    yaxis={'title': 'Population'},
                    hovermode='closest'
                )
            }
        )
        return graph
    else:
        print('loading error')
        return None


if __name__ == "__main__":

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
    app.layout = load_layout()
    app.run_server()
