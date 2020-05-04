import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from scipy.signal import savgol_filter
import utilities as utl
from init import app
import numpy as np
from navbar import Navbar

nav = Navbar()


def load_date_list():
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = utl.load_data_2()
    return date_list


def load_case_list(region="US"):
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = utl.load_data_2()
    return data_list_confirmed[region]

def load_case_list_regions():  # return data for all retions-of-interest
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = utl.load_data_2()
    return data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest


def load_death_list(region="US"):
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = utl.load_data_2()
    return data_list_deaths[region]


def compute_increase_rate(data_list_confirmed, region='US'):
    # print("...computing increase rate...\n")
    # data_list_confirmed, date_list = utl.load_data_3(region)
    A = data_list_confirmed[region]
    # print("confirmed data is ", A)
    # rate = [(A[k + 1] - A[k]) / A[k] * 100 for k in range(0, len(A) - 1)]
    rate = np.diff(np.array(A))
    return rate

def load_date_list_2(region='US'):
    data_list_confirmed, date_list = utl.load_data_3(region)
    # print("region is {}, date from {} ~ {}".format(
    #     region, date_list[0], date_list[-1]))

    return date_list[1:]


def smooth_list(l, window=3, poly=1):
    return savgol_filter(l, window, poly)


def update_increase_rate_row(region="US"):
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = load_case_list_regions()
    confirmed_inc = compute_increase_rate(data_list_confirmed, region)
    death_inc = compute_increase_rate(data_list_deaths, region)
    return dbc.Row([dbc.Col([
        dcc.Graph(id='increase rate',
                  figure={"data": [{"x": date_list[1:],  # load_date_list_2(region),
                                    "y": confirmed_inc,
                                    'mode': "lines+markers",
                                    'name': region + " Confirmed"},
                                   {"x": date_list[1:],  # load_date_list_2(region),
                                    "y": smooth_list(confirmed_inc, 5, 2),
                                    'mode': 'lines+markers',
                                    'name': 'Smoothed Confirmed'},
                                   {"x": date_list[1:],  # load_date_list_2(region),
                                    "y": death_inc,
                                    'mode': "lines+markers",
                                    'name': region + " Death"},
                                   {"x": date_list[1:],  # load_date_list_2(region),
                                    "y": smooth_list(death_inc, 5, 2),
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


def confirmed_vs_death(data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest, region="US"):
    return dbc.Row([dbc.Col([
        dcc.Graph(id='confirmed cases vs death',
                  figure={"data": [{"x": date_list,  # load_date_list(),
                                    "y": data_list_confirmed['US'],  # load_case_list("US"),
                                    'mode': "lines+markers",
                                    'name': 'Confirmed Cases'},
                                   {"x": date_list,  # load_date_list(),
                                    "y": data_list_deaths,  # load_death_list("US"),
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


def load_table(data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest):
    return html.Div(
        [
            dbc.Row([
                dbc.Col(
                    html.Div(html.H2("CADSEA - Covid 19 Analysis")),
                )
            ]
            ),
            # dbc.Row([
            #     dbc.Col([
            #         html.Div(html.P("Dashboard in Python")),
            #     ]
            #     )
            # ]
            # ),
            dbc.Row([
                dbc.Col([
                    html.Span(
                        [
                            html.H5(
                                ["US Confirmed", dbc.Badge(data_list_confirmed["US"][-1], color="warning", className="ml-1"),
                                 dbc.Badge("+" + str(data_list_confirmed["US"][-1] - data_list_confirmed["US"][-2]),
                                           color="warning",
                                           className="ml-1")]),
                        ]
                    ),
                ]
                ),
                dbc.Col([
                    html.Span(
                        [
                            html.H5(["US Death", dbc.Badge(data_list_deaths["US"][-1], color="danger", className="ml-1"),
                                     dbc.Badge("+" + str(data_list_deaths["US"][-1] - data_list_deaths["US"][-2]),
                                               color="danger",
                                               className="ml-1")]),
                        ]
                    ),
                ]
                ),
            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Span(
                        [
                            html.H5(
                                ["Italy Confirmed",
                                 dbc.Badge(data_list_confirmed["Italy"][-1], color="warning", className="ml-1"),
                                 dbc.Badge("+" + str(data_list_confirmed["Italy"][-1] - data_list_confirmed["Italy"][-2]),
                                           color="warning",
                                           className="ml-1")]),
                        ]
                    ),
                ]
                ),
                dbc.Col([
                    html.Span(
                        [
                            html.H5(["Italy Death",
                                     dbc.Badge(data_list_deaths["Italy"][-1], color="danger", className="ml-1"),
                                     dbc.Badge("+" + str(data_list_deaths["Italy"][-1] - data_list_deaths["Italy"][-2]),
                                               color="danger",
                                               className="ml-1")]),
                        ]
                    ),
                ]
                ),

            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Span(
                        [
                            html.H5(
                                ["China Confirmed",
                                 dbc.Badge(data_list_confirmed["China"][-1], color="warning", className="ml-1"),
                                 dbc.Badge("+" + str(data_list_confirmed["China"][-1] - data_list_confirmed["China"][-2]),
                                           color="warning",
                                           className="ml-1")]),
                        ]
                    ),
                ]
                ),
                dbc.Col([
                    html.Span(
                        [
                            html.H5(["China Death",
                                     dbc.Badge(data_list_deaths["China"][-1], color="danger", className="ml-1"),
                                     dbc.Badge("+" + str(data_list_deaths["China"][-1] - data_list_deaths["China"][-2]),
                                               color="danger",
                                               className="ml-1")]),
                        ]
                    ),
                ]
                ),

            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Span(
                        [
                            html.H5(
                                ["Canada Confirmed",
                                 dbc.Badge(data_list_confirmed["Canada"][-1], color="warning", className="ml-1"),
                                 dbc.Badge("+" + str(data_list_confirmed["Canada"][-1] - data_list_confirmed["Canada"][-2]),
                                           color="warning",
                                           className="ml-1")]),
                        ]
                    ),
                ]
                ),
                dbc.Col([
                    html.Span(
                        [
                            html.H5(["Canada Death",
                                     dbc.Badge(data_list_deaths["Canada"][-1], color="danger", className="ml-1"),
                                     dbc.Badge("+" + str(data_list_deaths["Canada"][-1] - data_list_deaths["Canada"][-2]),
                                               color="danger",
                                               className="ml-1")]),
                        ]
                    ),
                ]
                ),

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
                ], width=2,
                )
            ]
            ),
        ],
        style={"border": "60px white solid"}  # TODO
    )


# define the body layout
def load_body(data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest):

    return html.Div(
        [
            dbc.Row([
                dbc.Col([
                    html.Div(confirmed_vs_death(data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest),
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
        style={"border": "20px white solid"}  # TODO
    )


@app.callback(
    Output('confirmed cases vs death', 'figure'),
    [Input('Region_of_interest', 'value')])
def update_figure(value):
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = load_case_list_regions()
    return {
        "data": [{"x": date_list,
                  "y": data_list_confirmed[value],
                  'mode': "lines+markers",
                  'name': 'Confirmed Cases'},
                 {"x": date_list,
                  "y": data_list_deaths[value],
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
def update_rate_figure(value):
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = load_case_list_regions()
    return {
        "data": [{"x": date_list,
                  "y": compute_increase_rate(data_list_confirmed, value),
                  'mode': "lines+markers",
                  'name': value + " Confirmed"},
                 {"x": date_list,
                  "y": smooth_list(compute_increase_rate(data_list_confirmed, value),
                                   9,
                                   2),
                  'mode': 'lines+markers',
                  'name': 'Smoothed Confirmed'},
                 {"x": date_list,
                  "y": compute_increase_rate(data_list_deaths, value),
                  'mode': "lines+markers",
                  'name': value + " Death"},
                 {"x": date_list,
                  "y": smooth_list(compute_increase_rate(data_list_deaths, value),
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
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = load_case_list_regions()
    layout = html.Div([
        nav,
        load_table(data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest),
        load_body(data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest),
    ])
    return layout


if __name__ == "__main__":
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
    app.layout = load_layout()
    app.run_server()
