import matplotlib.pyplot as plt; plt.ion()
import numpy as np
import networkx as nx



def create_network(specs = {'n01': 2, 'n0': 3, 'n1': 5}):
    net = nx.empty_graph()
    net.add_edge(0, 1)

    n = 2
    for i in range(n, n + specs['n01']):
        net.add_edge(0, i)
        net.add_edge(1, i)

    n = len(net)
    for i in range(n, n + specs['n0']):
        net.add_edge(0, i)

    n = len(net)
    for i in range(n, n + specs['n1']):
        net.add_edge(1, i)

    return net


def edge_colormaps(net):
    e_col = []
    for e in net.edges():
        d = net.degree(e[1])
        if d > 2:
            e_col.append('coral')
        elif d == 2:
            e_col.append('lightskyblue')
        else:
            e_col.append('lightgrey')
    return e_col


def net_colormap(specs):
    n_size = [200] * 2
    n_col = ['orangered'] * 2

    n_size += [150] * specs['n01']
    n_col += ['royalblue'] * specs['n01']

    n_size += [100] * (specs['n0'] + specs['n1'])
    n_col += ['slategrey'] * (specs['n0'] + specs['n1'])
    return n_size, n_col


def plot_net(specs = {'n01': 5, 'n0': 36, 'n1': 50}, adj=.2):
    net = create_network(specs)
    fig, ax = plt.subplots()
    node_size, node_color = net_colormap(specs)
    edge_color = edge_colormaps(net)
    pos = nx.spring_layout(net)
    pos = adjust_pos(pos, specs, adj)
    nx.draw_networkx(net, pos=pos, ax=ax, node_size=node_size, node_color=node_color, edge_color=edge_color, width=2, with_labels=False)


def adjust_pos(pos, specs, adj=.1):
    ctr = (pos[0] + pos[1]) / 2
    v0 = pos[0] - pos[1]
    v = np.array([v0[0], -v0[1]])
    for node in range(2, 2 + specs['n01']):
        if np.dot(pos[node], v) >= np.dot(ctr, v):
            pos[node] += adj * v
        else:
            pos[node] += -adj * v
    return pos



