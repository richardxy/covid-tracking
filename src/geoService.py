import math
import numpy as np

from uszipcode import SearchEngine
from uszipcode import model


from src.dataService import dataServiceCSBS as CSBS


class geoClass:

    class point:
        def __init__(self, lng, lat):
            self.lng = lng
            self.lat = lat

        def dict(self):
            d = {'lon': self.lng, 'lat': self.lat}
            return d

        def __str__(self):
            return 'lat:{}, lng:{}'.format(self.lat, self.lng)

    class rect:
        def __init__(self, x0, x1, y0, y1):
            self.left = x0
            self.right = x1
            self.top = y1
            self.bottom = y0

        def __str__(self):
            return '{} <--> {}, {} ^ {}'.format(
                self.left, self.right, self.bottom, self.top)

    __instance = None

    def __new__(cls):

        if not cls.__instance:
            cls.__instance = object.__new__(cls)
            cls.__instance.__init__()
            print('.... geoClass Created, id:{}'.format(id(cls.__instance)))

        return cls.__instance

    def __init__(self):

        search = SearchEngine()

        self.defaultZipcodeInfo = search.by_zipcode('21029')

        self.defaultRadius = 70

        self.ds = CSBS()
        print('.... geoClass Initialized, id(self.ds):{}'.format(id(self.ds)))

    def geo_layout(self, title, projection='natural earth'):  # Confirmed Total
        layout = dict(title='CoronaVirus {} as of {}'.format(title, str(date.today())),
                      height=700,
                      colorbar=True,
                      geo=dict(
            # scope='usa',
            projection=projection,  # dict( type='albers usa' ),
            showland=True,
            landcolor="rgb(250, 250, 250)",
            subunitcolor="rgb(217, 217, 217)",
            countrycolor="rgb(217, 217, 217)",
            countrywidth=0.5,
            subunitwidth=0.5
        ),
        )
        return layout

    def distance(self, origin, destination):
        lat1, lon1 = origin
        lat2, lon2 = destination
        radius = 3959  # mile

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = radius * c
        return d

    def rectByZipAndEdge(self, center, edge):
        # return the rect of lat/lon range
        lat, lng = center.lat, center.lng
        radius = 3959  # mile
        delta = float(edge) * 180 / radius / math.pi
        return self.rect(lng - delta, lng + delta, lat - delta, lat + delta)

    def zipcodeInfo(self, zipcode):

        search = SearchEngine()

        # get info for the given zip code
        zipcode_info = search.by_zipcode(zipcode)
        if not zipcode_info.zipcode:
            zipcode_info = self.defaultZipcodeInfo

        return self.point(lng=zipcode_info.lng, lat=zipcode_info.lat)

    def get_regions(self, zipcode='22030', radius=100):

        zipcode_info = self.zipcodeInfo(zipcode)

        res = SearchEngine().query(
            lat=zipcode_info.lat,
            lng=zipcode_info.lng,
            radius=radius,
            # model.Zipcode.median_household_income,
            sort_by=model.SimpleZipcode.population,
            ascending=False,
            returns=12,
        )  # get 12 biggest cities around 100 miles around the given zip code

        counties = set([(r.post_office_city, r.state, r.county) for r in res])

        return counties

    def calc_zoom(self, rect):
        # https://stackoverflow.com/questions/46891914/control-mapbox-extent-in-plotly-python-api
        width_y = rect.top - rect.bottom
        width_x = rect.right - rect.left
        zoom_y = -1.446 * math.log(width_y) + 7.2753
        zoom_x = -1.415 * math.log(width_x) + 8.7068
        return min(round(zoom_y, 2), round(zoom_x, 2))

    def geo_data(self, category, zipcode, radius):

        self.center = self.zipcodeInfo(zipcode)
        self.radius = radius

        self.rectArea = self.rectByZipAndEdge(self.center, radius)

        self.zoom = self.calc_zoom(self.rectArea)

        df = self.ds.dataSet[category].copy()

        df = df[['County_Name', 'State_Name', 'Latitude',
                 'Longitude', self.ds.latest_date]].fillna(0)
        df = df.rename(columns={self.ds.latest_date: category})

        df['size'] = df.apply(lambda r: np.log10(r[category] + 1), axis=1)

        df = df[df['Longitude'] <= self.rectArea.right]
        df = df[df['Longitude'] >= self.rectArea.left]
        df = df[df['Latitude'] <= self.rectArea.top]
        df = df[df['Latitude'] >= self.rectArea.bottom]

        return df

    def search_by_zipcode(self, zipcode="21029"):

        df_conf = self.ds.dataSet["Confirmed"]

        date_cols = [c for c in df_conf.columns if '/20' in c]

        nomi = pgeocode.Nominatim('us')
        zipinfo = nomi.query_postal_code(zipcode)
        dist_vals = df_conf.apply(row_dist, axis=1, zipinfo=zipinfo)
        df_local = df_conf[dist_vals < 100]
        df_local = df_local[date_cols]

        return df_local, date_cols

    # def fetch_lat_long_by_name(self):
    #     from geopy.geocoders import Nominatim
    #     import pandas as pd

    #     df_conf = pd.read_csv('../data/time_series_19-covid-Confirmed.csv')
    #     df_us = df_conf[df_conf['Country/Region'] == 'US']
    #     location_names = df_us['Province/State'].values.tolist()
    #     geolocator = Nominatim(user_agent="corona")
    #     unqueried = location_names
    #     list_dict = []
    #     while len(unqueried) > 0:
    #         for l in location_names:
    #             try:
    #                 if ',' not in l:
    #                     location = geolocator.geocode(l + ' State')
    #                 else:
    #                     location = geolocator.geocode(l)
    #                 unqueried.remove(l)
    #                 list_dict.append({'Province/State': l, 'Lat': location.latitude, 'Long': location.longitude})
    #             except BaseException:
    #                 print('time out for querying %s' % l)

    #     latlong_df = pd.DataFrame.from_records(list_dict)
    #     latlong_df.to_csv('../data/lat_long_by_loc_name.csv')


if __name__ == '__main__':
    geo = geoClass()

    zipcode = '21029'
    radius = '100'

    center = geo.zipcodeInfo(zipcode)
    rectArea = geo.rectByZipAndEdge(center, radius)
    zoom = geo.calc_zoom(rectArea)

    print('center: {}'.format(center))
    print('rectArea: {}'.format(rectArea))
    print('zoom: {}'.format(zoom))
