from datetime import date, datetime, timedelta
from statistics import mean

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from plotly.validators.scatter.marker import SymbolValidator
from scipy.signal import savgol_filter

from init import app
from navbar import Navbar
from src.dataService import dataServiceCSBS as CSBS


def best_fit_slope(xs, ys):
    """Compute the best fit slope to (xs, ys)

    Arguments:
        xs {numpy array} -- x axis
        ys {numpy array} -- y axis

    Returns:
        float -- best fit slope
    """
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    return m


def get_slopes(df_Confirmed, country, column_name='Country/Region', patten='/20', strp='%Y-%m-%d'):
    base_date = '2020-01-15'
    # countries = df_Confirmed[column_name].unique().tolist()
    # country_slopes = []
    # for country in countries:

    us_df = df_Confirmed[df_Confirmed[column_name] == country].fillna(0)
    dates = [c for c in us_df.columns if patten in c]
    us_df_s = us_df[dates].sum(0)
    us_cases = [us_df_s[c] for c in dates]
    df_tmp = pd.DataFrame({'date': dates, 'cases': us_cases})
    df_tmp['date'] = pd.to_datetime(df_tmp['date'])
    df_tmp['days_since_basedate'] = (
        df_tmp['date'] - datetime.strptime(base_date, strp)).dt.days
    # df_t = df_tmp[df_tmp['days_since_basedate']>=0]
    # ax = sns.lineplot(x= 'days_since_basedate', y= 'cases',data=df_t)
    # plt.show()

    window_size = 5
    slopes = []
    five_days = []
    values = []
    for i in range(0, len(dates) - window_size):
        tmp_dt = df_tmp['date'].iloc[i:i+window_size].tolist()
        tmp_x = df_tmp['days_since_basedate'].iloc[i:i+window_size].tolist()
        tmp_y = us_cases[i:i+window_size]
        xs = np.array(tmp_x, dtype=np.float64)
        ys = np.array(tmp_y, dtype=np.float64)
        mean_y = np.average(ys)
        slopes.append(best_fit_slope(xs, ys)/mean_y if mean_y > 0 else 0)
        five_days.append(tmp_dt[-1])
        values.append(ys)
        # df_5d = pd.DataFrame({'fivedate':five_days,'slope':slopes})
    rec = {'Country': country, 'five-date': five_days,
           'slope': slopes, 'value': values}
    # country_slopes.append({'Country':country, 'five-date':five_days,'slope':slopes,'value':values})
    return rec


raw_symbols = SymbolValidator().values

nav = Navbar()


header = html.H3(
    'Select states: '
)


def load_region_options():
    ds = CSBS()
    ds.dataSet['Confirmed'] = ds.dataSet['Confirmed'].fillna(0)
    ds.dataSet['Deaths'] = ds.dataSet['Deaths'].fillna(0)
    # data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = utl.load_data_2()
    region_of_interest = ds.regions()
    region_options = [{'label': x, 'value': x} for x in region_of_interest]

    return region_options


region_defaults = ['Virginia', 'New York',
                   'District of Columbia', 'Maryland']  # region_of_interest[:7]

# return options, defaults


# region_options, region_defaults = load_options()

dropdown = html.Div(dcc.Dropdown(
    id='Region_of_interest',
    options=load_region_options(),
    value=region_defaults,
    multi=True

))

radioItems = html.Div([
    html.Label('Time Window', id='time_range'),
    dcc.RadioItems(
        id='tab1_TimeWindow',
        options=[
            {'label': 'All', 'value': 'ALL'},
            {'label': 'Last Month', 'value': 'MON'},
            {'label': 'Last Two Weeks', 'value': 'WEEKS'},
            {'label': 'Last Week', 'value': 'WEEK'},
        ],
        value='ALL'
    )
])


confirmedGraph = html.Div(id='comfirmed_region_graph',
                          children=[],
                          )
deathsGraph = html.Div(id='deaths_region_graph',
                       children=[],
                       )

confirmedDailyIncGraph = html.Div(id='comfirmed_dailyinc_graph',
                                  children=[],
                                  )
deathsDailyIncGraph = html.Div(id='deaths_dailyinc_graph',
                               children=[],
                               )

confirmedIncRateGraph = html.Div(id='comfirmed_incrate_graph',
                                 children=[],
                                 )
deathsIncRateGraph = html.Div(id='deaths_incrate_graph',
                              children=[],
                              )


def App():
    layout = html.Div([
        nav,
        dbc.Row([dbc.Col(header, align='center', width=2),
                 dbc.Col(dropdown, width=9)]),

        dbc.Row([dbc.Col(radioItems, width=6)]),
        dbc.Row([dbc.Col(confirmedGraph, width=6)]),
        dbc.Row([dbc.Col(deathsGraph, width=6)]),
        dbc.Row([dbc.Col(confirmedDailyIncGraph, width=6)]),
        dbc.Row([dbc.Col(deathsDailyIncGraph, width=6)]),
        dbc.Row([dbc.Col(confirmedIncRateGraph, width=6)]),
        dbc.Row([dbc.Col(deathsIncRateGraph, width=6)]),

    ])
    return layout


@app.callback(
    [Output(component_id='comfirmed_region_graph', component_property='children'),
     Output(component_id='deaths_region_graph', component_property='children'),
     Output(component_id='comfirmed_dailyinc_graph',
            component_property='children'),
     Output(component_id='deaths_dailyinc_graph',
            component_property='children'),
     Output(component_id='comfirmed_incrate_graph',
            component_property='children'),
     Output(component_id='deaths_incrate_graph',
            component_property='children'),
     Output(component_id='time_range', component_property='children'),
     ],
    [Input(component_id='tab1_TimeWindow', component_property='value'),
     Input('Region_of_interest', 'value')])
def update_graph(date_window_option, region_of_interest):
    ds = CSBS()
    dt_range = ds.date_range_str(date_window_option)

    confirmed = ds.refresh_category(
        'Confirmed',
        date_window_option,
        region_of_interest)
    deaths = ds.refresh_category(
        'Deaths',
        date_window_option,
        region_of_interest)

    return plot_figure(confirmed, 'Confirmed', dt_range), \
        plot_figure(deaths, 'Deaths', dt_range), \
        plot_inc_number(confirmed, 'Confirmed', dt_range), \
        plot_inc_number(deaths, 'Deaths', dt_range), \
        plot_increase(ds, 'Confirmed', region_of_interest), \
        plot_increase(ds, 'Deaths', region_of_interest), \
        'Time Window:{}'.format(dt_range)


def plot_figure(df, category, dt_range):

    graph = dcc.Graph(
        figure={
            'data': df,  # data,
            'layout': go.Layout(
                title='{} cases over date<br>{}'.format(
                    category, dt_range),
                yaxis={'title': 'Population'},
                hovermode='closest'
            )
        }
    )
    return graph


def plot_increase(ds, category, region_of_interest):
    df_ds = ds.dataSet[category]

    # df_ds['county_state'] = df_ds['County_Name'] + ', ' + df_ds['State_Name']
    # county_list = df_ds['State_Name'].unique().tolist()
    ret = []
    for state in region_of_interest:
        county_slopes = get_slopes(
            df_ds, state, column_name='State_Name', patten='2020-', strp='%Y-%m-%d')
        ret.append({'x': county_slopes['five-date'], 'y': county_slopes['slope'],
                    'name': state, 'mode': 'lines+markers'})
    graph = dcc.Graph(
        figure={
            'data': ret,  # data,
            'layout': go.Layout(
                title='{} increase rate over date<br>'.format(
                    category),
                yaxis={'title': 'Population'},
                hovermode='closest'
            )
        }
    )
    return graph

# def compute_increase_number(df_Confirmed, country, column_name, pattern='2020-'):
#     return np.diff(np.array(confirmed[1]['y']))


def smooth_list(l, window=3, poly=1):
    return savgol_filter(l, window, poly)


def plot_inc_number(confirmed, category, dt_range):
    # df_ds = ds.dataSet[category]

    ret = []
    for state_data in confirmed:
        state_incs = np.diff(np.array(state_data['y']))
        state_smooth = smooth_list(state_incs, 5, 2)
        ret.append({'x': state_data['x'], 'y': state_incs,
                    'name': '{} Daily Increase'.format(state_data['name']), 'mode': 'lines+markers'})
        ret.append({'x': state_data['x'], 'y': state_smooth,
                    'name': '{} Daily Increase Smoothed'.format(state_data['name']), 'mode': 'lines+markers'})
    graph = dcc.Graph(
        figure={
            'data': ret,  # data,
            'layout': go.Layout(
                title='Number of {} Daily Increase at {}<br>'.format(
                    category, state_data['name']),
                yaxis={'title': 'Number'},
                hovermode='closest'
            )
        }
    )
    return graph
