import math

import dash_table
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc

from dash.dependencies import Output, Input, State

from src.geoService import geoClass
from src.newsService import newsClass

from init import app
import utilities as utl

# Navbar
from navbar import Navbar

from plotly.validators.scatter.marker import SymbolValidator

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
    value='21029',
    id='zipcodeInput',
    style=dict(display='flex', justifyContent='center', width='50'),
)

radiusInput = dcc.Input(
    placeholder='Enter a radius(miles)...',
    type='text',
    value='100',
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
                    ], style={'display':'flex','vertical-align': 'right'}),
                 ),
            ]),
            dbc.Row([html.Div([
                html.H1('empty row'), 
                ],
                style={'color': 'white'})]),
            dbc.Row([
                dbc.Col([
                    dbc.Badge(html.Div(html.H6("Confirmed Cases")),pill=True, color='primary'),
                ]
                ),
                dbc.Col([
                    dbc.Badge(html.Div(html.H6("Death Cases")),pill=True, color='danger'),

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
            dbc.Row([dbc.Col([html.H3("Related news"),], width={'size':3, 'offset':5})]),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            localNews
                        ], width = {'size':6,'offset':3},
                    )
                ], 
            )
        ],
        style={"border": "20px white solid"}  ##TODO
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


print('.... Page_search loaded, id(geoSvr):{}, id(newsSvr):{}'.format(
    id(geoSvr), id(newsSvr)))
