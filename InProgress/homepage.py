import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from scipy.signal import savgol_filter
import utilities as utl
from init import app

from navbar import Navbar

nav = Navbar()


def load_date_list():
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = utl.load_data_2()
    return date_list


def load_case_list(region="US"):
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = utl.load_data_2()
    return data_list_confirmed[region]


def load_death_list(region="US"):
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = utl.load_data_2()
    return data_list_deaths[region]


def compute_increase_rate(region='US'):
    print("...computing increase rate...\n")
    data_list_confirmed, date_list = utl.load_data_3(region)
    A = data_list_confirmed
    # print("confirmed data is ", A)
    rate = [(A[k + 1] - A[k]) / A[k] * 100 for k in range(0, len(A) - 1)]
    return rate


def compute_death_increase_rate(region='US'):
    print("...computing death increase rate...\n")
    data_list_deaths, date_list = utl.load_data_4(region)
    A = data_list_deaths
    # print("confirmed data is ", A)
    rate = [(A[k + 1] - A[k]) / A[k] * 100 if A[k] > 0 else 0 for k in range(0, len(A) - 1)]
    return rate


def load_date_list_2(region='US'):
    data_list_confirmed, date_list = utl.load_data_3(region)
    print("region is {}, date from {} ~ {}".format(
        region, date_list[0], date_list[-1]))

    return date_list[1:]


def smooth_list(l, window=3, poly=1):
    return savgol_filter(l, window, poly)


def update_increase_rate_row(region="US"):
    return dbc.Row([dbc.Col([
        dcc.Graph(id='increase rate',
                  figure={"data": [{"x": load_date_list_2(region),
                                    "y": compute_increase_rate(region),
                                    'mode': "lines+markers",
                                    'name': region + " Confirmed"},
                                   {"x": load_date_list_2(region),
                                    "y": smooth_list(compute_increase_rate(region),
                                                     9,
                                                     2),
                                    'mode': 'lines+markers',
                                    'name': 'Smoothed Confirmed'},
                                   {"x": load_date_list_2(region),
                                    "y": compute_death_increase_rate(region),
                                    'mode': "lines+markers",
                                    'name': region + " Death"},
                                   {"x": load_date_list_2(region),
                                    "y": smooth_list(compute_death_increase_rate(region),
                                                     9,
                                                     2),
                                    'mode': "lines+markers",
                                    'name': 'Smoothed Death'},
                                   ],
                          'layout': {
                              'title': region + " Daily Confirmed and Death Increase Rate",
                              'background': 0000,
                          }
                          }),
    ],
        width="auto",
    ),
    ])


def confirmed_vs_death(region="US"):
    return dbc.Row([dbc.Col([
        dcc.Graph(id='confirmed cases vs death',
                  figure={"data": [{"x": load_date_list(),
                                    "y": load_case_list("US"),
                                    'mode': "lines+markers",
                                    'name': 'Confirmed Cases'},
                                   {"x": load_date_list(),
                                    "y": load_death_list("US"),
                                    'mode': "lines+markers",
                                    'name': 'Death Cases'},
                                   ],
                          'layout': {
                              'title': region + " Daily Confirmed and Death Chart",
                              'background': 0000,
                          }
                          }),
    ],
        width="auto",
    ),
    ])


def load_table():
    return html.Div(
        [
            dbc.Row([
                dbc.Col([
                    html.Div(html.H1("CADSEA - Covid 19 Analysis")
                             ),
                ]
                )
            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Div(html.P("Dashboard in Python")
                             ),
                ]
                )
            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Span(
                        [
                            html.H5(["US Death", dbc.Badge(load_death_list("US")[-1], color="danger", className="ml-1"),
                                     dbc.Badge("+" + str(load_death_list("US")[-1] - load_death_list("US")[-2]),
                                               color="danger",
                                               className="ml-1")]),
                        ]
                    ),
                ]
                ),
                dbc.Col([
                    html.Span(
                        [
                            html.H5(
                                ["US Confirmed", dbc.Badge(load_case_list("US")[-1], color="warning", className="ml-1"),
                                 dbc.Badge("+" + str(load_case_list("US")[-1] - load_case_list("US")[-2]),
                                           color="warning",
                                           className="ml-1")]),
                        ]
                    ),
                ]
                )
            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Span(
                        [
                            html.H5(["Italy Death",
                                     dbc.Badge(load_death_list("Italy")[-1], color="danger", className="ml-1"),
                                     dbc.Badge("+" + str(load_death_list("Italy")[-1] - load_death_list("Italy")[-2]),
                                               color="danger",
                                               className="ml-1")]),
                        ]
                    ),
                ]
                ),
                dbc.Col([
                    html.Span(
                        [
                            html.H5(
                                ["Italy Confirmed",
                                 dbc.Badge(load_case_list("Italy")[-1], color="warning", className="ml-1"),
                                 dbc.Badge("+" + str(load_case_list("Italy")[-1] - load_case_list("Italy")[-2]),
                                           color="warning",
                                           className="ml-1")]),
                        ]
                    ),
                ]
                )
            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Span(
                        [
                            html.H5(["China Death",
                                     dbc.Badge(load_death_list("China")[-1], color="danger", className="ml-1"),
                                     dbc.Badge("+" + str(load_death_list("China")[-1] - load_death_list("China")[-2]),
                                               color="danger",
                                               className="ml-1")]),
                        ]
                    ),
                ]
                ),
                dbc.Col([
                    html.Span(
                        [
                            html.H5(
                                ["China Confirmed",
                                 dbc.Badge(load_case_list("China")[-1], color="warning", className="ml-1"),
                                 dbc.Badge("+" + str(load_case_list("China")[-1] - load_case_list("China")[-2]),
                                           color="warning",
                                           className="ml-1")]),
                        ]
                    ),
                ]
                )
            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Span(
                        [
                            html.H5(["Canada Death",
                                     dbc.Badge(load_death_list("Canada")[-1], color="danger", className="ml-1"),
                                     dbc.Badge("+" + str(load_death_list("Canada")[-1] - load_death_list("Canada")[-2]),
                                               color="danger",
                                               className="ml-1")]),
                        ]
                    ),
                ]
                ),
                dbc.Col([
                    html.Span(
                        [
                            html.H5(
                                ["Canada Confirmed",
                                 dbc.Badge(load_case_list("Canada")[-1], color="warning", className="ml-1"),
                                 dbc.Badge("+" + str(load_case_list("Canada")[-1] - load_case_list("Canada")[-2]),
                                           color="warning",
                                           className="ml-1")]),
                        ]
                    ),
                ]
                )
            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Div(html.H5("                     ")
                             ),
                ]
                )
            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Div(html.H6("Choose a Region/Country: ")
                             ),
                ]
                )
            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Div([dcc.Dropdown(
                        id='Region_of_interest',
                        options=[
                            {'label': 'US', 'value': 'US'},
                            {'label': 'Italy', 'value': 'Italy'},
                            {'label': 'Canada', 'value': 'Canada'},
                            {'label': 'China', 'value': 'China'}
                        ],
                        value='US',
                        clearable=False,
                    ),
                    ],

                    )
                ]
                )
            ]
            ),
        ],
        style={"border": "60px white solid"}  ##TODO
    )


# define the body layout
def load_body():
    return html.Div(
        [
            dbc.Row([
                dbc.Col([
                    html.Div(confirmed_vs_death('US'),
                             ),
                ]
                ),
                dbc.Col([
                    html.Div(update_increase_rate_row('US'),
                             ),

                ],
                ),

            ]
            )
        ],
        style={"border": "20px white solid"}  ##TODO
    )


@app.callback(
    Output('confirmed cases vs death', 'figure'),
    [Input('Region_of_interest', 'value')])
def update_figure(value):
    return {
        "data": [{"x": load_date_list(),
                  "y": load_case_list(value),
                  'mode': "lines+markers",
                  'name': 'Confirmed Cases'},
                 {"x": load_date_list(),
                  "y": load_death_list(value),
                  'mode': "lines+markers",
                  'name': 'Death Cases'},
                 ],
        'layout': {
            'title': value + " Daily Confirmed and Death Chart"
        }
    }


@app.callback(
    Output('increase rate', 'figure'),
    [Input('Region_of_interest', 'value')])
def update_figure(value):
    return {
        "data": [{"x": load_date_list_2(value),
                  "y": compute_increase_rate(value),
                  'mode': "lines+markers",
                  'name': value + " Confirmed"},
                 {"x": load_date_list_2(value),
                  "y": smooth_list(compute_increase_rate(value),
                                   9,
                                   2),
                  'mode': 'lines+markers',
                  'name': 'Smoothed Confirmed'},
                 {"x": load_date_list_2(value),
                  "y": compute_death_increase_rate(value),
                  'mode': "lines+markers",
                  'name': value + " Death"},
                 {"x": load_date_list_2(value),
                  "y": smooth_list(compute_death_increase_rate(value),
                                   9,
                                   2),
                  'mode': "lines+markers",
                  'name': 'Smoothed Death'},
                 ],
        'layout': {
            'title': value + " Daily Confirmed and Death Increase Rate"
        }
    }


def load_layout():
    layout = html.Div([
        nav,
        load_table(),
        load_body(),
    ])
    return layout


if __name__ == "__main__":
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
    app.layout = load_layout()
    app.run_server()
