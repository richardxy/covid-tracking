import EoN
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import random
N = 100000
conn_num = 50
print('generating graph G with {} nodes'.format(N))

G = nx.fast_gnp_random_graph(N, conn_num/(N-1))
#We add random variation in the rate of leaving exposed class
#and in the partnership transmission rate.
#There is no variation in recovery rate.
node_attribute_dict = {node: random.uniform(0.6,0.99) for node in G.nodes()}
edge_attribute_dict = {edge: random.uniform(0.6, 0.8) for edge in G.edges()}
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
H.add_edge('E', 'I', rate = 0.1, weight_label='expose2infect_weight')
H.add_edge('I', 'R', rate = 0.01)
J = nx.DiGraph() #for the induced transitions
J.add_edge(('I', 'S'), ('I', 'E'), rate = 0.001,weight_label='transmission_weight')
J.add_edge(('S','I'), ('S','E'), rate = 0.7,weight_label='susep_weight')
J.add_edge(('E','I'), ('E','S'), rate = 0.5,weight_label='expose_weight')
IC = defaultdict(lambda: 'S')
for node in range(200):
    IC[node] = 'I'
return_statuses = ('S', 'E', 'I', 'R')
print('doing Gillespie simulation')
t, S, E, I, R = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses,tmax = float('Inf'))


import EoN
import networkx as nx
import matplotlib.pyplot as plt
import random

import pickle

#set up the code to handle constant transmission rate
#with fixed recovery time.
def trans_time_fxn(source, target, rate):
    return random.expovariate(rate)

def rec_time_fxn(node,D):
    return D

results = []
N=1000000
conn_num_s =  [5, 10, 50]

D_s = [1, 2, 4]
tau_s = [0.1, 0.3, 0.7]
initial_inf_count = 100

total = len(D_s) * len(tau_s) * len(conn_num_s)
cnt = 0
for conn_num in conn_num_s:
    for D in D_s:
        for tau in tau_s:

            G = nx.fast_gnp_random_graph(N, conn_num/(N-1.))
            t, S, I, R = EoN.fast_nonMarkov_SIR(G,
                                    trans_time_fxn=trans_time_fxn,
                                    rec_time_fxn=rec_time_fxn,
                                    trans_time_args=(tau,),
                                    rec_time_args=(D,),
                                    initial_infecteds = range(initial_inf_count))
            data_dict = {'rec_time':D, 'trans_time':tau, 't':t, 'I':I,'S':S,'R':R, 'population':N, 'conn_num':conn_num}
            results.append(data_dict)
            cnt += 1
            print('Done {} out of {}'.format(cnt, total))


print('done with simulation, now plotting')
plt.plot(t, S, label = 'Susceptible')
# plt.plot(t, E, label = 'Exposed')
plt.plot(t, I, label = 'Infected')
plt.plot(t, R, label = 'Recovered')
plt.xlabel('$t$')
plt.ylabel('Simulated numbers')
plt.legend()
plt.show()


pickle_out = open("simulation_results.pickle","wb")
pickle.dump(results, pickle_out)
