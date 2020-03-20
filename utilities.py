#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 23:26:04 2020

@author: chaowu
"""

import numpy as np
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
from datetime import date

import plotly.express as px

from flask import jsonify
import pgeocode
import math



colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

def dataSource(category:int) -> str:
    '''
    Return the URL for the given category, which is one of the following:
    Confirmed, Deaths, Recovered
    
    Return None if the given parameter is none of the above three.

    '''
    base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"

    dataSources = {'Confirmed':'time_series_19-covid-Confirmed.csv',
                   'Deaths':'time_series_19-covid-Deaths.csv',
                   'Recovered':'time_series_19-covid-Recovered.csv'}
    
    fn = dataSources.get(category)
    
    return base_url+fn if fn else fn 

def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 3959 # mile

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

def row_dist(r, zipinfo):
    dist = distance((zipinfo['latitude'], zipinfo['longitude']), (r['Lat'],r['Long']))
    return dist 

def config_geo_layout(px):

    current_time = str(date.today())

    # print('reading the csv')
    df_conf = pd.read_csv(dataSource('Confirmed'))
    date_cols = [c for c in df_conf.columns if '/20' in c]
    df_conf['total'] = df_conf[date_cols].sum(axis=1)
    df_conf = df_conf[df_conf['total'] > 0]

    df_deaths = pd.read_csv(dataSource('Deaths'))
    df_deaths['total'] = df_deaths[date_cols].sum(axis=1)
    df_deaths = df_deaths[df_deaths['total'] > 0]

    # print('done reading the csv file')

    def get_text(r):
        # print(r['Province/State'])
        if str(r['Province/State']) != 'nan':
            return r['Province/State'] + '<br>' + 'Confirmed: ' + str(r['total'])
        else:
            return r['Country/Region'] + '<br>' + 'Confirmed: ' + str(r['total'])

    df_conf['text'] = df_conf.apply(get_text, axis=1)
    median_val = df_conf['total'].median()

    # scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    #     [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]
    scl = [[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [0.5, "rgb(70, 100, 245)"],
           [0.6, "rgb(90, 120, 245)"], [0.7, "rgb(106, 137, 247)"], [1, "rgb(240, 210, 250)"]]

    df_deaths['text'] = df_deaths.apply(get_text, axis=1)

    scl = [[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [0.5, "rgb(70, 100, 245)"],
           [0.6, "rgb(90, 120, 245)"], [0.7, "rgb(106, 137, 247)"], [1, "rgb(220, 220, 220)"]]

    median_val_d = df_deaths['total'].median()

    data = [dict(
        type='scattergeo',
        # locationmode =  #'USA-states',
            locations="iso_alpha",
            lon=df_conf['Long'],
            lat=df_conf['Lat'],
            text=df_conf['text'],
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.8,
                reversescale=True,
                autocolorscale=False,
                symbol='circle',
                line=dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
                # colorscale = scl,
                color_continuous_scale=px.colors.diverging.BrBG,
                color_continuous_midpoint=median_val,
                cmin=0,
                color=df_conf['total'],
                cmax=df_conf['total'].max(),
                colorbar=dict(
                    title="Confirmed Total"
                )
            ))]

    layout = dict(
        title='CoronaVirus Confirmed Total of ' + current_time,
        height=700,
        colorbar=True,
        geo=dict(
            # scope='usa',
            projection='natural earth',  # dict( type='albers usa' ),
            showland=True,
            landcolor="rgb(250, 250, 250)",
            subunitcolor="rgb(217, 217, 217)",
            countrycolor="rgb(217, 217, 217)",
            countrywidth=0.5,
            subunitwidth=0.5
        ),
    )

    return data, layout


def load_data():

    df_Confirmed = pd.read_csv(dataSource("Confirmed"))
    df_Deaths = pd.read_csv(dataSource("Deaths"))
    df_Recovered = pd.read_csv(dataSource("Recovered"))

    #countries = df_Confirmed['Country/Region'].unique()

    date_list = df_Confirmed.columns.to_list()
    date_list = date_list[34:]

    region_of_interest = ['US', 'Germany', 'Italy', 'United Kingdom', 'Canada', 'Iran','Spain']

    def update_number_by_region(df=df_Confirmed):
        data_list = []
        for region in region_of_interest:
            #print("region is ", region)
            df_1 = df[df['Country/Region'] == region]
            df_1 = df_1.fillna(0)

            confirmed_number = list(np.sum(np.array(df_1[date_list]), axis=0))
            confirmed_number = [int(x) for x in confirmed_number]
            data_list.append({'x': date_list, 'y': confirmed_number, 'mode': 'lines+markers', 'name': region})

        return data_list

    data_list_confirmed = update_number_by_region(df_Confirmed)
    data_list_deaths = update_number_by_region(df_Deaths)
    data_list_recovered = update_number_by_region(df_Recovered)

    # print(data_list_confirmed)

    return data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest


def organize_figure_structure(data):

    figure_data = {
        'data': data,
        'layout': {
            'plot_bgcolor': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {'color': colors['text']}
        }
    }

    return figure_data

def tab_1_layout():

    # load data on the fly
    print("loading data on the fly n tab_1_layout...")
    data_list_confirmed, data_list_deaths, data_list_recovered, date_list, region_of_interest = load_data()

    return html.Div([
        html.H3(children='Confirmed case',
                style={'textAlign': 'center', 'color': colors['text']}
                ),

        dcc.Graph(
            id='Graph1',
            figure=organize_figure_structure(data_list_confirmed)
        ),

        html.H3(children='Death case',
                style={'textAlign': 'center', 'color': colors['text']}
                ),

        dcc.Graph(
            id='Graph2',
            figure=organize_figure_structure(data_list_deaths)
        ),

        html.H3(children='Recovered case',
                style={'textAlign': 'center', 'color': colors['text']}
                ),

        dcc.Graph(
            id='Graph3',
            figure=organize_figure_structure(data_list_recovered)
        )
    ])


def tab_2_layout():

    # load the data and layout at fly
    data, layout = config_geo_layout(px)
    fig = dict(data=data, layout=layout)

    data_d, layout_d = config_geo_layout(px)
    fig_d = dict(data=data_d, layout=layout_d)

    return html.Div([
        html.H3('Geo Distribution'),

        dcc.Graph(id='virus_c_graph', figure=fig),
        dcc.Graph(id='virus_d_graph', figure=fig_d)
    ])

    
    
def search_by_zipcode(zipcode="21029"):
    
    df_conf = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')#('time_series_19-covid-Confirmed.csv')
    date_cols = [c for c in df_conf.columns if '/20' in c]
    date_cols = date_cols[-1]
    
    nomi = pgeocode.Nominatim('us')
    zipinfo = nomi.query_postal_code(zipcode)
    dist_vals = df_conf.apply(row_dist, axis = 1, zipinfo = zipinfo)
    df_local = df_conf[dist_vals< 100]
    df_local = df_local[date_cols]
    
    return df_local


def tab_3_layout():
 
    # display search result

    return html.Div([
        html.H3(children = 'Search by zipcode (US only)',
                style={'textAlign': 'center', 'color': colors['text']}),
                
        html.Button(id='submit-button', n_clicks=0, children='Submit',
                    style={'textAlign': 'center', 'color': colors['text']}),

    ])

    
def tab_4_layout():
 
    # display search result

    return html.Div([
        html.H3('under construction'),
    ])
   
    
if __name__ == "__main__":
    df_conf = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')#('time_series_19-covid-Confirmed.csv')
    date_cols = [c for c in df_conf.columns if '/20' in c]
    print('done reading data into dataframe')

    df_local = search_by_zipcode()
    
    print(df_local)