import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from scipy.signal import savgol_filter
import utilities as utl

from navbar import Navbar
nav = Navbar()


def load_date_list():
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = utl.load_data_2()
    return date_list


def load_case_list(region="US"):
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = utl.load_data_2()
    return data_list_confirmed[region]


def compute_increase_rate(region='US'):
    print("...computing increase rate...\n")
    data_list_confirmed, date_list = utl.load_data_3(region)
    A = data_list_confirmed
    # print("confirmed data is ", A)
    rate = [(A[k + 1] - A[k]) / A[k] * 100 for k in range(0, len(A) - 1)]
    return rate


def load_date_list_2(region='US'):
    data_list_confirmed, date_list = utl.load_data_3(region)
    print("region is {}, date from {} ~ {}".format(
        region, date_list[0], date_list[-1]))

    return date_list[1:]


def smooth_list(l, window=3, poly=1):
    return savgol_filter(l, window, poly)


def update_increase_rate_row(region="US"):
    return dbc.Row([dbc.Col([html.H2(region + " daily increase rate"),
                             dcc.Graph(figure={"data": [{"x": load_date_list_2(region),
                                                         "y": compute_increase_rate(region),
                                                         'mode': "lines+markers",
                                                         'name': region},
                                                        {"x": load_date_list_2(region),
                                                         "y": smooth_list(compute_increase_rate(region),
                                                                          9,
                                                                          2),
                                                         'mode': 'lines+markers',
                                                         'name': 'Smoothed'},
                                                        ],
                                               "layout": utl.layout}),
                             ]),
                    ])


def update_increase_list(region_list=['US', 'Italy', 'Spain']):
    row_list = []
    for region in region_list:
        row_list.append(update_increase_rate_row[region])

    return row_list

# define the body layout

def load_body():
    return dbc.Container([dbc.Row([dbc.Col([html.H1("Covid 19 Status"),
                                        html.P("""
                            This coronavirus spreading is changing the world dramatically.
                            """),
                                        html.A(html.Button('Details',
                                                           className='google_result'),
                                               href='https://www.worldometers.info/coronavirus/country/us/'),
                                        ],
                                       md=4,
                                       ),
                               dbc.Col([html.H2("Confirmed cases"),
                                        dcc.Graph(figure={"data": [{"x": load_date_list(),
                                                                    "y": load_case_list("US"),
                                                                    'mode': "lines+markers",
                                                                    'name': 'US'},
                                                                   {"x": load_date_list(),
                                                                    "y": load_case_list("Italy"),
                                                                    'mode': "lines+markers",
                                                                    'name': 'Italy'},
                                                                   {"x": load_date_list(),
                                                                    "y": load_case_list("Spain"),
                                                                    'mode': "lines+markers",
                                                                    'name': 'Spain'},
                                                                   ],
                                                          "layout": utl.layout}),
                                        ],
                                       md=10,
                                       ),
                               ]),
                      update_increase_rate_row('US'),
                      update_increase_rate_row('Italy'),
                      update_increase_rate_row('Spain'),
                      ],
                     className="mt-4",
                     )

def load_layout():
    layout = html.Div([
        nav,
        load_body(),
    ])
    return layout


if __name__ == "__main__":

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
    app.layout = load_layout()
    app.run_server()
