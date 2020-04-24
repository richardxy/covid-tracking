import utilities as util 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime 
import numpy as np 
from statistics import mean
from src.dataService import dataServiceCSBS as CSBS

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    return m

def get_slopes(df_Confirmed, column_name='Country/Region',patten='/20',strp = '%Y-%m-%d'):
    countries = df_Confirmed[column_name].unique().tolist()
    country_slopes = []
    for country in countries:
        us_df = df_Confirmed[df_Confirmed[column_name]==country]
        dates = [c for c in us_df.columns if patten in c]
        us_cases = [us_df[c].iloc[0] for c in us_df.columns if patten in c]
        df_tmp = pd.DataFrame({'date':dates,'cases':us_cases})
        df_tmp['date'] = pd.to_datetime(df_tmp['date'])
        df_tmp['days_since_basedate'] = (df_tmp['date'] - datetime.strptime(base_date,strp)).dt.days
        # df_t = df_tmp[df_tmp['days_since_basedate']>=0]
        # ax = sns.lineplot(x= 'days_since_basedate', y= 'cases',data=df_t)
        # plt.show()

        window_size = 5
        slopes = []
        five_days = []
        values = []
        for i in range(0, len(dates) - window_size):
            tmp_x = df_tmp['days_since_basedate'].iloc[i:i+window_size].tolist()
            tmp_y = us_cases[i:i+window_size]
            xs = np.array(tmp_x, dtype=np.float64)
            ys = np.array(tmp_y, dtype=np.float64)
            slopes.append(best_fit_slope(xs,ys)/ys[-1] if ys[-1]>0 else 0)
            five_days.append(xs[-1])
            values.append(ys)
        # df_5d = pd.DataFrame({'fivedate':five_days,'slope':slopes})    

        country_slopes.append({'Country':country, 'five-date':five_days,'slope':slopes,'value':values})
    return country_slopes

base_date = '2020-02-15'
df_Confirmed = pd.read_csv(util.dataSource("Confirmed"))
countries = df_Confirmed['Country/Region'].unique().tolist()
country_slopes = []
for country in countries:
    us_df = df_Confirmed[df_Confirmed['Country/Region']==country]
    dates = [c for c in us_df.columns if '/20' in c]
    us_cases = [us_df[c].iloc[0] for c in us_df.columns if '/20' in c]
    df_tmp = pd.DataFrame({'date':dates,'cases':us_cases})
    df_tmp['date'] = pd.to_datetime(df_tmp['date'])
    df_tmp['days_since_basedate'] = (df_tmp['date'] - datetime.strptime(base_date,'%Y-%m-%d')).dt.days
    # df_t = df_tmp[df_tmp['days_since_basedate']>=0]
    # ax = sns.lineplot(x= 'days_since_basedate', y= 'cases',data=df_t)
    # plt.show()

    window_size = 5
    slopes = []
    five_days = []
    values = []
    for i in range(0, len(dates) - window_size):
        tmp_x = df_tmp['days_since_basedate'].iloc[i:i+window_size].tolist()
        tmp_y = us_cases[i:i+window_size]
        xs = np.array(tmp_x, dtype=np.float64)
        ys = np.array(tmp_y, dtype=np.float64)
        slopes.append(best_fit_slope(xs,ys)/ys[-1] if ys[-1]>0 else 0)
        five_days.append(xs[-1])
        values.append(ys)
    # df_5d = pd.DataFrame({'fivedate':five_days,'slope':slopes})    

    country_slopes.append({'Country':country, 'five-date':five_days,'slope':slopes,'value':values})

ax = sns.lineplot(x= c['five-date'], y=c['slope'])
plt.show()

ds = CSBS()
df_ds = ds.dataSet['Confirmed']
df_ds['county_state'] = df_ds['County_Name'] + ', ' + df_ds['State_Name']
county_list = df_ds['county_state'].unique().tolist()
county_slopes = get_slopes(df_ds,column_name='county_state',patten='2020-',strp = '%Y-%m-%d')
ax = sns.lineplot(x= county_slopes[0]['five-date'], y=county_slopes[0]['slope'])
####################################
import networkx as nx
import EoN
import matplotlib.pyplot as plt
N = 10**6 #number of individuals
kave = 15 #expected number of partners
print('generating graph G with {} nodes'.format(N))
G = nx.fast_gnp_random_graph(N, kave/(N-1)) #Erdo''s-Re'nyi graph
rho = 0.005 #initial fraction infected
tau = 1.5 #transmission rate
gamma = 1 #recovery rate

print('doing event-based simulation')
t1, S1, I1, R1 = EoN.fast_SIR(G, tau, gamma, rho=rho,return_full_data=True)
#instead of rho, we could specify a list of nodes as initial_infecteds, or
#specify neither and a single random node would be chosen as the index case.
print('doing Gillespie simulation')

t2, S2, I2, R2 = EoN.Gillespie_SIR(G, tau, gamma, rho=rho, return_full_data=True)
print('done with simulations, now plotting')
plt.plot(t1, I1, label = 'fast_SIR')
plt.plot(t2, I2, label = 'Gillespie_SIR')
plt.xlabel('$t$')
plt.ylabel('Number infected')
plt.legend()
plt.show()


################
import EoN
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import random
N = 100000
print('generating graph G with {} nodes'.format(N))
G = nx.fast_gnp_random_graph(N, 5./(N-1))
#We add random variation in the rate of leaving exposed class
#and in the partnership transmission rate.
#There is no variation in recovery rate.
node_attribute_dict = {node: 0.5+random.random() for node in G.nodes()}
edge_attribute_dict = {edge: 0.5+random.random() for edge in G.edges()}
nx.set_node_attributes(G, values=node_attribute_dict,
name='expose2infect_weight')
nx.set_edge_attributes(G, values=edge_attribute_dict,
name='transmission_weight')
#
#These individual and partnership attributes will be used to scale
#the transition rates. When we define \texttt{H} and \texttt{J}, we provide the name
#of these attributes.

#More advanced techniques to scale the transmission rates are shown in
#the online documentation
H = nx.DiGraph() #For the spontaneous transitions
H.add_node('S') #This line is actually unnecessary.
H.add_edge('E', 'I', rate = 0.6, weight_label='expose2infect_weight')
H.add_edge('I', 'R', rate = 0.1)
J = nx.DiGraph() #for the induced transitions
J.add_edge(('I', 'S'), ('I', 'E'), rate = 0.1,
weight_label='transmission_weight')
IC = defaultdict(lambda: 'S')
for node in range(200):
    IC[node] = 'I'
return_statuses = ('S', 'E', 'I', 'R')
print('doing Gillespie simulation')
t, S, E, I, R = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses,tmax = float('Inf'))

print('done with simulation, now plotting')
plt.plot(t, S, label = 'Susceptible')
plt.plot(t, E, label = 'Exposed')
plt.plot(t, I, label = 'Infected')
plt.plot(t, R, label = 'Recovered')
plt.xlabel('$t$')
plt.ylabel('Simulated numbers')
plt.legend()
plt.show()