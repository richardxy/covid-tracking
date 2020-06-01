import numpy as np
import pandas as pd
import os
import numpy as np
from statistics import mean
from datetime import datetime, date, timedelta
from re import findall
import requests
# import plotly.express as px


class dataServiceCSSE:
    '''
    Instance varible list:
    today
    categories
    region_of_interest
    dataSet
    all_date_range
    date_list

    '''

    def __init__(self):

        self.today = date.today()  # check wether date changed
        self.categories = ['Confirmed', 'Deaths', 'Recovered']
        self.region_of_interest = ['US', 'Germany', 'Italy', 'United Kingdom', 'Canada', 'Iran', 'Spain', 'China', 'Japan', 'Singapore', 'Australia'
                                   ]
        self.init_datasSet()

    def regions(self):
        return self.region_of_interest

    def init_datasSet(self):
        # Fetch data
        self.dataSet = {}
        for category in self.categories:
            self.dataSet[category] = pd.read_csv(
                self.dataSource(category),
                error_bad_lines=False)

        self.all_date_range = self.init_date_range(self.dataSet[category])

    def dataSource(self, category: int) -> str:
        '''
        Return the URL for the given category, which is one of the following:
        Confirmed, Deaths, Recovered

        Return None if the given parameter is none of the above three.

        '''
        base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"

        dataSources = {'Confirmed': 'time_series_19-covid-Confirmed.csv',
                       'Deaths': 'time_series_19-covid-Deaths.csv',
                       'Recovered': 'time_series_19-covid-Recovered.csv'}

        fn = dataSources.get(category)

        return base_url+fn if fn else fn

    def init_date_range(self, df):
        # if ALL_DATES already populated, use it directdly in the same day

        # otherwiese populate it
        date_list = df.columns.to_list()

        return date_list[4:]

    def date_range(self, date_window_option='ALL'):
        # ALL, MONS, MON, WEEKS, WEEK
        # format- m/d/yy

        if date_window_option == 'ALL':
            return self.all_date_range

        dtDelta = {'MONS': 60, 'MON': 30, 'WEEKS': 14, 'WEEK': 7}

        ret = []
        for d in range(dtDelta[date_window_option], -1, -1):
            dt = self.today - timedelta(days=d)
            ret.append('{}/{}/{}'.format(dt.month, dt.day, dt.year % 100))

        return ret

    def refresh_data(self, date_window_option, region_of_interest):
        # refresh data as per :category, date list and region of interest

        date_list = self.date_range(date_window_option)
        ret = []
        for category in ['Confirmed', 'Deaths', 'Recovered']:
            ret.append(self.refresh_category(category,
                                             date_list,
                                             region_of_interest
                                             ))

        return ret[0], ret[1], ret[2], ' {} ~ {}'.format(date_list[0], date_list[-1])

    def refresh_category(self, category, date_list, region_of_interest):

        df = self.dataSet[category]

        ret = []
        for region in region_of_interest:
            #print("region is ", region)
            df_1 = df[df['Country/Region'] == region]
            df_1 = df_1.fillna(0)

            counts = list(np.sum(np.array(df_1[date_list]), axis=0))
            counts = [int(x) for x in counts]
            ret.append({'x': date_list, 'y': counts,
                        'mode': 'lines+markers', 'name': region})

        return ret


class dataServiceCSBS(object):
    '''
    Instance varible list:
    today
    categories
    region_of_interest
    dataSet
    all_date_range
    date_list

    '''

    __instance = None
    today = None

    def __new__(cls, *agrs, **kwargs):

        if not cls.__instance:
            print('__new__ not _instance ')
            cls.__instance = object.__new__(cls, *agrs, **kwargs)
            cls.__instance.__init__()

        return cls.__instance

    def __init__(self):

        if self.today == date.today():
            return

        print('__init__')
        self.today = date.today()  # check wether date changed
        self.categories = ['Confirmed', 'Deaths']
        self.all_date_range = []
        self.df_columns = ['County_Name', 'State_Name',
                           'Confirmed', 'New',
                           'Deaths', 'Fatality_Rate',
                           'Latitude', 'Longitude',
                           'Last_Update']
        self.init_datasSet()
        dataServiceCSBS.__instance = self

    def regions(self):
        return self.region_of_interest

    def init_regions(self, Confirmed):
        #
        sample_date = self.all_date_range[-1]
        s = Confirmed[['State_Name', sample_date]].groupby(['State_Name'])\
            .sum().sort_values(sample_date, ascending=False)
        self.region_of_interest = s.index.tolist()  # [0:17]

    def read_csv(self, fn):
        '''
        Instead of fetch from url, this function will try to loacte
        the csv file from local drive
        '''

        csv_base_url = 'https://raw.githubusercontent.com/tomquisel/covid19-data/master/'

        local_data_folder = './data/CSBS/'

        if not os.path.exists(local_data_folder):
            os.mkdir(local_data_folder)

        local_data_fn = os.path.join(
            local_data_folder, fn[9:])  # remove prefix data/csv/

        try:
            Confirmed = pd.read_csv(local_data_fn, error_bad_lines=False)
        except BaseException:
            Confirmed = pd.read_csv(csv_base_url + fn, error_bad_lines=False)
            Confirmed.to_csv(local_data_fn, index=False)

        return Confirmed

    def init_datasSet(self):

        git_master_url = "https://github.com/tomquisel/covid19-data/tree/master/"

        csvs = findall('data/csv/2020-0[0-9-]{4}.csv',
                       requests.get(git_master_url+'data/csv/').text)

        self.all_date_range = []
        Confirmed = self.read_csv(csvs[0])
        Confirmed[['Latitude', 'Longitude']] = Confirmed[[
            'Latitude', 'Longitude']] .astype(float)

        #Confirmed.set_index(['County_Name','State_Name'], inplace=True)

        Deaths = Confirmed.copy()

        for csv in csvs:

            data = self.read_csv(csv)
            date = csv[9:-4]  # 2020-mm-dd

            # print(Confirmed.columns)
            c = data[['County_Name', 'State_Name', 'Confirmed']]
            c.columns = ['County_Name', 'State_Name', date]

            Confirmed = pd.merge(Confirmed, c,
                                 how='outer',
                                 on=['County_Name', 'State_Name'])

            d = data[['County_Name', 'State_Name', 'Death']]
            d.columns = c.columns
            Deaths = pd.merge(Deaths, d, how='outer',
                              on=['County_Name', 'State_Name'])

            self.all_date_range.append(date)

        self.dataSet = {}
        self.dataSet[self.categories[0]] = Confirmed
        self.dataSet[self.categories[1]] = Deaths

        self.latest_date = self.all_date_range[-1]
        self.date_list = self.all_date_range

        self.init_regions(Confirmed)

    def date_range(self, date_window_option='ALL'):
        # ALL, MONS, MON, WEEKS, WEEK
        # format- m/d/yy

        if date_window_option == 'ALL':
            self.date_list = self.all_date_range
            return self.all_date_range

        dtDelta = {'MONS': 60, 'MON': 30, 'WEEKS': 14, 'WEEK': 7}

        ret = []
        for d in range(dtDelta[date_window_option], -1, -1):
            dt = self.today - timedelta(days=d)
            ymd = '{}-{:02}-{:02}'.format(dt.year, dt.month, dt.day, dt.year)
            if ymd in self.all_date_range:
                ret.append(ymd)

        self.date_list = ret

        return ret

    def refresh_category(self, category: str, date_window_option, region_of_interest):
        # refresh data as per :category, date list and region of interest
        df = self.dataSet[category].copy()

        self.date_list = self.date_range(date_window_option)

        ret = []
        for region in region_of_interest:
            #print("region is ", region)
            df_1 = df[df['State_Name'] == region]
            df_1 = df_1.fillna(0)
            # print(df_1.columns)
            counts = list(np.sum(np.array(df_1[self.date_list]), axis=0))
            counts = [int(x) for x in counts]
            ret.append({'x': self.date_list, 'y': counts,
                        'name': region, 'mode': 'lines+markers'})

        return ret

    def refresh_county_state_category(self, category: str, date_window_option, region_of_interest):
        # refresh data as per :category, date list and region of interest
        df = self.dataSet[category].copy()

        self.date_list = self.date_range(date_window_option)

        ret = []
        for county_state in region_of_interest: # region_of_interest is a list of "county, state"
            #print("region is ", region)
            region = [s.strip() for s in county_state.split(',')]
            if region[1] == 'District of Columbia':
                region[0] = 'Washington'
            df_1 = df[(df['State_Name'] == region[1])&(df['County_Name'] == region[0])]
            if df_1.shape[0]== 0:
                df_1 = df[(df['State_Name'] == region[1])&(df['County_Name'] == region[0].split(' ')[0].strip())]
            
            if df_1.shape[0] > 0 and df_1.iloc[0,-1]>0:
                df_1 = df_1.fillna(0)
            
                counts = df_1[self.date_list].values[0].tolist()
                counts = [int(x) for x in counts]
                ret.append({'x': self.date_list, 'y': counts,
                            'name': county_state, 'mode': 'lines+markers'})

        return ret

    def get_increase_rate(self, category: str, date_window_option, region_of_interest):
        # refresh data as per :category, date list and region of interest
        df = self.dataSet[category].copy()

        self.date_list = self.date_range(date_window_option)

        ret = []
        for region in region_of_interest:
            #print("region is ", region)
            df_1 = df[df['State_Name'] == region]
            df_1 = df_1.fillna(0)

            counts = list(np.sum(np.array(df_1[self.date_list]), axis=0))
            counts = [int(x) for x in counts]
            ret.append({'x': self.date_list, 'y': counts,
                        'name': region, 'mode': 'lines+markers'})

        return ret

    def date_range_str(self, date_window_option='ALL'):

        self.date_list = self.date_range(date_window_option)

        return '{}~{}'.format(self.date_list[0], self.date_list[-1])


#ds = dataServiceCSSE()


#ds = dataServiceCSBS()

# if __name__ == '__main__':
#     print(ds.columns)
