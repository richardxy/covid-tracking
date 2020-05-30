import math

import dash_table
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc

from dash.dependencies import Output, Input, State
from datetime import date, datetime, timedelta
from src.geoService import geoClass
from src.newsService import newsClass

from init import app
import utilities as utl
import numpy as np 
import pandas as pd 
# Navbar
from navbar import Navbar
from statistics import mean
from plotly.validators.scatter.marker import SymbolValidator
from scipy.signal import savgol_filter
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
def get_slopes(county_rec):
    base_date = '2020-01-15'
    strp='%Y-%m-%d'
    # countries = df_Confirmed[column_name].unique().tolist()
    # country_slopes = []
    # for country in countries:

    # us_df = df_Confirmed[df_Confirmed[column_name] == country].fillna(0)
    dates = county_rec['x']
    # us_df_s = us_df[dates].sum(0)
    us_cases = county_rec['y']
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
    rec = {'County': county_rec['name'], 'five-date': five_days,
           'slope': slopes, 'value': values}
    # country_slopes.append({'Country':country, 'five-date':five_days,'slope':slopes,'value':values})
    return rec


def plot_ts_figure(df, category, dt_range):

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


def plot_increase(df, category, dt_range):
    # df_ds = ds.dataSet[category]

    # # df_ds['county_state'] = df_ds['County_Name'] + ', ' + df_ds['State_Name']
    # # county_list = df_ds['State_Name'].unique().tolist()
    ret = []
    for county_rec in df:
        county_slopes = get_slopes(county_rec)
        ret.append({'x': county_slopes['five-date'], 'y': county_slopes['slope'],
                    'name': county_rec['name'], 'mode': 'lines+markers'})
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
        # state_smooth = smooth_list(state_incs, 5, 2)
        ret.append({'x': state_data['x'], 'y': state_incs,
                    'name': '{} Daily Increase'.format(state_data['name']), 'mode': 'lines+markers'})
        # ret.append({'x': state_data['x'], 'y': state_smooth,
        #             'name': '{} Daily Increase Smoothed'.format(state_data['name']), 'mode': 'lines+markers'})
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

def plot_figure(category, zipcode, radius):
    df = geoSvr.geo_data(category, zipcode, radius)

    fig = px.scatter_mapbox(df,
                            lat='Latitude', lon='Longitude',
                            hover_name='County_Name',
                            hover_data=[category],
                            # hovertemplate = '{hover_name}<br><br>'+category+':%{hover_data}',
                            # title="Custom layout.hoverlabel formating",
                            size='size',
                            # color_discrete_sequence=['fuchsia'],
                            zoom=geoSvr.zoom,
                            center=geoSvr.center.dict(),
                            height=420
                            )

    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(margin={'r': 10, 't': 10, 'l': 10, 'b': 10, })
    fig.update_layout(autosize=True)
    fig.update_layout(hoverlabel=dict())

    graph = dcc.Graph(
        figure=fig

    )

    figs[category] = graph

    return graph


def data_table(category, zipcode, radius):
    df = geoSvr.geo_data(category, zipcode, radius)
    df['county_state'] = df['County_Name']+', '+df['State_Name']
    df = df[['county_state', category]]
    return dash_table.DataTable(
        # id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'center'
            } for c in df.columns
        ],
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_as_list_view=True,
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_table={
            'maxHeight': '600px',
            'overflowY': 'scroll'
        },
    )
raw_symbols = SymbolValidator().values

# navigation bar
nav = Navbar()
geoSvr = geoClass()
newsSvr = newsClass()

figs = {}  # save curent fig, incase invalid zipcode inputed

header = html.H3(
    'Choose a zipcode and find nearby status.'
)

zipcodeInput = dcc.Input(
    placeholder='Enter a zipcode...',
    type='text',
    value='22030',
    id='zipcodeInput',
    style=dict(display='flex', justifyContent='center', width='50'),
)

radiusInput = dcc.Input(
    placeholder='Enter a radius(miles)...',
    type='text',
    value='50',
    id='radiusInput',
    style=dict(display='flex', justifyContent='center', width='30'),
)

inputDiv = html.Div([
    html.P('ZipCode:'),
    zipcodeInput,
    html.P(' Radius:'),
    radiusInput,
    html.P('miles'),
    html.Button(id='submit_zipcode_radius', n_clicks=0, children='GO'),
    html.P(id='inputMsg', children='', style={'color': 'red'})
], style=dict(display='flex'))

confirmedNearby = html.Div(id='confirmedNearby', )

deathsNearby = html.Div(id='deathsNearby', )

confirmedTable = html.Div(id='confirmedTable', )

deathsTable = html.Div(id='deathsTable', )

localNews = html.Div(id='localNews')

dropdown = html.Div(dcc.Dropdown(
    id='county_Region_of_interest',
    # options=load_region_options(),
    # value=region_defaults,
    multi=True

))

radioItems = html.Div([
    html.Label('Time Window', id='county_time_range'),
    dcc.RadioItems(
        id='county_tab1_TimeWindow',
        options=[
            {'label': 'All', 'value': 'ALL'},
            {'label': 'Last Month', 'value': 'MON'},
            {'label': 'Last Two Weeks', 'value': 'WEEKS'},
            {'label': 'Last Week', 'value': 'WEEK'},
        ],
        value='ALL'
    )
])
confirmedGraph = html.Div(id='county_comfirmed_region_graph',
                          children=[],
                          )
deathsGraph = html.Div(id='county_deaths_region_graph',
                       children=[],
                       )

confirmedDailyIncGraph = html.Div(id='county_comfirmed_dailyinc_graph',
                                  children=[],
                                  )
deathsDailyIncGraph = html.Div(id='county_deaths_dailyinc_graph',
                               children=[],
                               )

confirmedIncRateGraph = html.Div(id='county_comfirmed_incrate_graph',
                                 children=[],
                                 )
deathsIncRateGraph = html.Div(id='county_deaths_incrate_graph',
                              children=[],
                              )

def load_body():
    return html.Div(
        [
            dbc.Row([
                dbc.Col([
                    html.Div(html.H1("Neighbourhood")),
                ]
                )
            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Div(html.P("Type your Zipcode and Find Nearby Status: "),
                             ),
                ]
                ),
                dbc.Col(
                    html.Div([
                        html.H6('ZipCode: '),
                        zipcodeInput,
                        html.H6(' Radius: '),
                        radiusInput,
                        html.H6('miles'),
                        html.Button(id='submit_zipcode_radius', n_clicks=0, children='GO'),
                        html.H6(id='inputMsg', children='', style={'color': 'red'})
                    ], style={'display': 'flex', 'vertical-align': 'right'}),
                ),
            ]),
            dbc.Row([dbc.Col(header, align='center', width=2),
                    dbc.Col(dropdown, width=9)]),

            dbc.Row([dbc.Col(radioItems, width=6)]),
            dbc.Row([dbc.Col(confirmedGraph, width=6),
                    dbc.Col(deathsGraph, width=6)]),
            dbc.Row([dbc.Col(confirmedDailyIncGraph, width=6),
                    dbc.Col(deathsDailyIncGraph, width=6)]),
            dbc.Row([dbc.Col(confirmedIncRateGraph, width=6),
                    dbc.Col(deathsIncRateGraph, width=6)]),
            
            dbc.Row([html.Div([
                html.H1('empty row'),
            ],
                style={'color': 'white'})]),
            dbc.Row([
                dbc.Col([
                    dbc.Badge(html.Div(html.H6("Confirmed Cases")), pill=True, color='primary'),
                ]
                ),
                dbc.Col([
                    dbc.Badge(html.Div(html.H6("Death Cases")), pill=True, color='danger'),

                ],
                ),

            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Div(confirmedNearby,
                             ),
                ]
                ),
                dbc.Col([
                    html.Div(deathsNearby,
                             ),

                ],
                ),

            ]
            ),
            dbc.Row([
                dbc.Col([
                    html.Div(confirmedTable,
                             ),
                ]
                ),
                dbc.Col([
                    html.Div(deathsTable,
                             ),

                ],
                ),

            ]
            ),
            dbc.Row([dbc.Col([html.H3("Related news"), ], width={'size': 3, 'offset': 5})]),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            localNews
                        ], width={'size': 6, 'offset': 3},
                    )
                ],
            )
        ],
        style={"border": "20px white solid"}  # TODO
    )


def App():
    layout = html.Div([
        nav,
        # header,
        # inputDiv,
        load_body(),
        # html.H3('Confirmed Cases'),
        # confirmedNearby,
        # html.H3("Deaths cases"),
        # deathsNearby,
        # confirmedTable,
        # deathsTable,
        html.H3("Related news"),
        # localNews
    ])
    return layout

@app.callback(
    [Output(component_id='county_comfirmed_region_graph', component_property='children'),
     Output(component_id='county_deaths_region_graph', component_property='children'),
     Output(component_id='county_comfirmed_dailyinc_graph',
            component_property='children'),
     Output(component_id='county_deaths_dailyinc_graph',
            component_property='children'),
     Output(component_id='county_comfirmed_incrate_graph',
            component_property='children'),
     Output(component_id='county_deaths_incrate_graph',
            component_property='children'),
     Output(component_id='county_time_range', component_property='children'),
     Output(component_id = 'county_Region_of_interest',component_property='options'),
     ],
    [Input(component_id='zipcodeInput', component_property='value'),
     Input('radiusInput', 'value'),
     Input(component_id='county_tab1_TimeWindow', component_property='value'),
     Input('county_Region_of_interest', 'value')])
def update_ts_graph(zipcode, radius, date_window_option='ALL', region_of_interest=[]):
    print('zipcode: {}'.format(zipcode))
    if len(zipcode) == 5:
        radius = float(radius)
        recs = geoSvr.search_by_zipcode('Confirmed',zipcode,radius) # list of (county,state)
        region_options = ['{}, {}'.format(r[0],r[1]) for r in recs]
        # region_defaults = ['{}, {}'.format(r[0],r[1]) for r in recs]
        ds = CSBS()
        dt_range = ds.date_range_str(date_window_option)

        if region_of_interest is None:
            region_of_interest = []
        # print('len of region_options:{}, of region_of_interest:{}'.format(region_options is None, region_of_interest is None))
        if set(region_of_interest).isdisjoint(region_options):
            region_of_interest = region_options

        confirmed = ds.refresh_county_state_category(
            'Confirmed',
            date_window_option,
            region_of_interest)
        deaths = ds.refresh_county_state_category(
            'Deaths',
            date_window_option,
            region_of_interest)

        return plot_ts_figure(confirmed, 'Confirmed', dt_range), \
            plot_ts_figure(deaths, 'Deaths', dt_range), \
            plot_inc_number(confirmed, 'Confirmed', dt_range), \
            plot_inc_number(deaths, 'Deaths', dt_range), \
            plot_increase(confirmed, 'Confirmed', dt_range), \
            plot_increase(deaths, 'Deaths', dt_range), \
            'Time Window:{}'.format(dt_range), \
            [{'label': x, 'value': x} for x in region_options]
        # region_defaults

@app.callback(
    [
        Output(component_id='confirmedNearby', component_property='children'),
        Output(component_id='deathsNearby', component_property='children'),
        Output(component_id='confirmedTable', component_property='children'),
        Output(component_id='deathsTable', component_property='children'),
        #  Output(component_id='localNews', component_property='children'),
        #  Output(component_id='inputMsg', component_property='children'),
    ],
    [Input(component_id='zipcodeInput', component_property='value'),
     Input('radiusInput', 'value')])
def update_map_and_news(zipcode, radius):
    # print('---------- type(newsSvr):{} ----------------'.format(type(newsSvr) ) )
    if len(zipcode) == 5:
        radius = float(radius)
    # plot_figure('Confirmed', zipcode, radius), \
        # plot_figure('Deaths', zipcode, radius), \
        return plot_figure('Confirmed', zipcode, radius), \
            plot_figure('Deaths', zipcode, radius), \
            data_table('Confirmed', zipcode, radius), \
            data_table('Deaths', zipcode, radius), \
            # newsSvr.show_news_list(zipcode, radius), \


@app.callback(
    # Output(component_id='confirmedNearby', component_property='children'),
    #  Output(component_id='deathsNearby', component_property='children'),
    #  Output(component_id='confirmedTable', component_property='children'),
    #  Output(component_id='deathsTable', component_property='children'),
    Output(component_id='localNews', component_property='children'),
    #  Output(component_id='inputMsg', component_property='children'),
    [Input('confirmedNearby', 'children')],
    [State(component_id='zipcodeInput', component_property='value'),
     State('radiusInput', 'value')])
def update_news(v, zipcode, radius):
    print('zipcode:{}, radius:{}'.format(zipcode, radius))
    return newsSvr.show_news_list(zipcode, float(radius))





print('.... Page_search loaded, id(geoSvr):{}, id(newsSvr):{}'.format(
    id(geoSvr), id(newsSvr)))
