import matplotlib.pyplot as plt; plt.ion()
import matplotlib.gridspec as gridspec
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
    n_size = [20] * 2
    n_col = ['orangered'] * 2

    n_size += [15] * specs['n01']
    n_col += ['royalblue'] * specs['n01']

    n_size += [10] * (specs['n0'] + specs['n1'])
    n_col += ['slategrey'] * (specs['n0'] + specs['n1'])
    return n_size, n_col


def plot_net(specs = {'n01': 5, 'n0': 36, 'n1': 50}, adj=.2, ax=None):
    net = create_network(specs)
    if ax is None:
        fig, ax = plt.subplots()
    node_size, node_color = net_colormap(specs)
    edge_color = edge_colormaps(net)
    pos = nx.spring_layout(net)
    pos = adjust_pos(pos, specs, adj)
    nx.draw_networkx(net, pos=pos, ax=ax, node_size=node_size, node_color=node_color, edge_color=edge_color, width=1, with_labels=False)
    ax.axis('off')



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



def plot_ts(times, obs_w, ax=None, obs_w_l=.15, levs=.1):
    ltimes = len(times)
    times = [obs_w[0]] + times + [obs_w[1]]
    times = [(t - obs_w[0])/(obs_w[1] - obs_w[0] + 0.) for t in times]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.8, 4))
    obs_w_l = obs_w_l
    levs = levs
    levels = [obs_w_l] +  [levs] * ltimes + [obs_w_l]

    stemc = ax.stem(times, levels, linefmt='k-', basefmt='k-')
    stemc[0].set_visible(False) #Remove 'dots' on top of stem plot

    ts_levs = (levs - obs_w_l)
    for i in range(1, len(stemc[1]) - 1):
        stemc[1][i].set_lw(.5)
    stemc[1][0].set_ydata([ts_levs, obs_w_l])
    stemc[1][0].set_ls('--') #Change color and style of first and last lines
    stemc[1][0].set_color('r')

    stemc[1][-1].set_ydata([ts_levs, obs_w_l])
    stemc[1][-1].set_ls('--')
    stemc[1][-1].set_color('r')

    stemc[2].set_ydata([levs/2] * 2) # move base to the middle

    ax.set_ylim((ts_levs, obs_w_l))
    ax.axis('off')


edge_values = {'b':{0:0, 2:0}, 'bt_n':{0:0, 2:0}, 't_stb':{0:0, 2:0}}

def plot_main_figure(edge_set, times_set, edge_values={}):
    obs_w = [1167609600, 1177977600]
    fig = plt.figure(tight_layout=True)
    widths = [1, 1, 1]
    heights = [2.5, 1, 2.5, 1]

    spec = gridspec.GridSpec(ncols=3, nrows=4, width_ratios=widths, height_ratios=heights, hspace=.25, wspace=.1)
    edges = get_reduced_edge_set(edge_set, edge_values)
    col_var = ['b', 'bt_n', 't_stb']
    for col in range(3):
        var = col_var[col]
        for row in [0, 2]: #row corresponds to weak=0/strong=2 case
            ax = fig.add_subplot(spec[row, col])
            edge = edges[var][row]

            edge_specs = edge_set[var][row][edge]
            plot_net(edge_specs, ax=ax)

            ax2 = fig.add_subplot(spec[row + 1, col])
            times = times_set[edge]
            plot_ts(times, obs_w, ax=ax2)

def get_reduced_edge_set(edge_set, edge_values={}):
    """
    Retuns the a dictionary where we select the nth (edge, times) pair for each [var][weak/strong] case. edges are sorted according to the first node id.
    """
    if not edge_values:
        edge_values = {k: {0: 0, 2: 0} for k in edge_set}
    es = {k: {} for k in edge_set}
    for k in edge_values:
        es[k][0] = sorted(edge_set[k][0])[edge_values[k][0]]
        es[k][2] = sorted(edge_set[k][2])[edge_values[k][2]]
    return es


def get_edge_set(df):
    """
    Given the dataframe for the whole data, gets possible edges for the main plot according to weak/strong values
    """
    vrs = ['b', 't_stb', 'bt_n', 'out_call_div']
    data_vrs = ['0', '1', 'w', 'ovrl', 'n_ij', 'deg_0', 'deg_1']
    edges = {v: {} for v in vrs}
    df_conds = (df.w > 35) & (df.w < 55) & (df.deg_0 < 80) & (df.deg_1 < 80) & (np.abs(df.deg_1 - df.deg_0) < 20)
    df = df[data_vrs + vrs][df_conds].sort_values('ovrl')

    for var in vrs:
        asc = True if var in ['t_stb', 'bt_n'] else False
        df2 = df[(df.ovrl < .03) & (df.ovrl > 0)].sort_values(var, ascending=asc)
        edges[var][0] = parse_edges(df2.quantile(np.linspace(.075, .15, 20)))
        df2 = df[(df.ovrl > .14) & (df.ovrl < .2)].sort_values(var, ascending=(not asc))
        edges[var][2] = parse_edges(df2.quantile(np.linspace(.075, .15, 20)))

    return edges


def parse_edges(edge):
    edges = {}
    for rowi in edge.iterrows():
        row = rowi[1]
        n01 = row['n_ij']
        n0 = row['deg_0'] - n01 - 1
        n1 = row['deg_1'] - n01 - 1
        edges[(int(row[0]), int(row[1]))] = {'n01': int(n01), 'n0': int(n0), 'n1': int(n1)}
    return edges


def get_times(edges, times_path='../full_run/times_dict.txt'):
    """
    Given the edge set from get_edge_set, gets the timestamps for those edges
    """
    import utils
    times_dict = {}
    edge_set = set([tuple(x) for sublist in edges.values() for y in sublist.values() for x in y])
    with open(times_path, 'r') as r:
        row = r.readline()
        while (row is not None) & (len(edge_set) > 0):
            e0, e1, times = utils.parse_time_line(row)
            edge = (e0, e1)
            if edge in edge_set:
                times_dict[edge] = times
                edge_set.remove(edge)
            row = r.readline()
    return times_dict


if __name__ == '__main__':
    import pandas as pd
    import pickle
    import os
    path = '/scratch/work/urenaj1/full/'

    df_path = path + 'full_df_paper.txt'
    times_path = path + 'times_dic.txt'

    edges_outpath = path + 'mainplot_edges.p'
    times_outpath = path + 'mainplot_times.p'

    if not os.path.exists(times_outpath):
        df = pd.read_csv(df_path, sep=' ')

        edges = get_edge_set(df)
        del df
        times = get_times(edges, times_path)

        pickle.dump(edges, open(edges_outpath, 'wb'))
        pickle.dump(times, open(times_outpath, 'wb'))
    else:
        edges = pickle.load(open(edges_outpath, 'rb'))
        times = pickle.load(open(times_outpath, 'rb'))

    edge_values = {'b':{0:0, 2:0}, 'bt_n':{0:0, 2:0}, 't_stb':{0:0, 2:0}}
    edge_values['b'][0] = 2
    edge_values['b'][2] = 5

