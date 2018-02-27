# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import pickle
import datetime as dt
#from netpython import *

# filter by city with: csvgrep -c 16 -m "Las Palmas" -e "utf-8" -d "," canarias/actives_cityloc_200701_200704.csv > canarias/palmas.csv
# loadNet
# x = netio.loadNet(path)
# pasos :   filter by city with csvgrep
#           write a pickle with the id's of people in delimitation with create_subset_dictionary
#           write a file with write_logs_subset
#           reorder tenerife ids
#               1. create dic with reorder_ids
#               2. use dic with reorder_log_ids

def write_logs_subset(log_path, dic_path, output_path, id_cols=[1,3], sep=' '):
    d = pd.read_pickle(dic_path)
    f = open(output_path, 'wb')
    with open(log_path) as r:
        row = r.readline()
        while row:
            rs = row.split(sep)
            if int(rs[id_cols[0]]) in d:
                f = open(output_path, 'a')
                f.write(row)
                f.close()
            elif int(rs[id_cols[1]]) in d:
                f = open(output_path, 'a')
                f.write(row)
                f.close()
            row=r.readline()
    r.close()

def reorder_log_ids(log_path, out_log_path, ids_path, id_cols=[1,3], sep=' '):
    id_dic = pd.read_pickle(ids_path)
    with open(log_path) as r:
        row = r.readline()
        id_0 = id_cols[0]
        id_1 = id_cols[1]
        while row:
            rs = row.split(sep)
            rs[id_0] = str(id_dic[int(rs[id_0])])
            rs[id_1] = str(id_dic[int(rs[id_1])])
            with open(out_log_path, 'a') as f:
                f.write(' '.join(rs))
            row = r.readline()


def reorder_ids(log_path, out_dic_path, id_cols=(1,3), sep=' '):
    d = {}
    ref = pd.read_csv(log_path, delimiter=sep, header=None)
    ref_set = set(ref[id_cols[0]]).union(set(ref[id_cols[1]]))
    i = 0
    for r in ref_set:
        d[r] = i
        i += 1
    pickle.dump(d, open(out_dic_path, 'wb'))


def write_edges(net, output_path):
    """
    Sort edges of a net and write a file with their sorted (or not) edges.
    If reverse=True, the biggest weights will go first.
    """
    f = open(output_path, "w+")
    for edge in net.edges:
        e = str(edge[0]) + " " + str(edge[1]) + " " + str(edge[2]) + "\n"
        f.write(e)
    f.close()

def write_dic(dic, output_path):
    f = open(output_path, "w+")
    for key, value in dic.iteritems():
        value = [str(v) for v in value]
        w = str(key[0]) + " " +  str(key[1]) + " " + " ".join(value) + "\n"
        f.write(w)
    f.close()

def create_subset_dictionary(ref_path='../data/mobile_network/canarias/actives_cityloc_200701_200704.csv', dic_path='', id_col=0):
    data = pd.read_csv(ref_path, header=None)
    d = set(data[id_col])
    pickle.dump(d, open(dic_path, 'wb'))

def add_provice_code(ref_path, data_path, data_col):
    prov_code = create_code_dic(ref_path)
    data = pd.read_csv(data_path)

def create_code_dic(ref_path):
    data = pd.read_csv(ref_path, sep=';', dtype=str)
    data.NPRO = data.NPRO.apply(lambda x: remove_accent(x))
    data.NPRO = data.NPRO.apply(lambda x: reorder_commas(x))
    prov_code = {}
    for row in data.iterrows():
        name = row[1].NPRO.split('/')
        for n in name:
            prov_code[n] = row[1].CPRO

    return prov_code

def remove_accent(x):
    subs = [('á', 'a'),
       ('é', 'e'),
       ('è','e'),
       ('í', 'i'),
       ('ó', 'o'),
       ('ú', 'u'),
       ('Á', 'A'),
       ('É', 'E'),
       ('Í', 'I'),
       ('Ó', 'O'),
       ('Ú', 'U'),
       ('ñ', 'n'),
       ('Ñ', 'n'),
       ('¥', 'n')]
    if pd.isnull(x):
        return None
    if isinstance(x, str):
        x_2 = x
        for tup_subs in subs:
            x_2 = re.sub(tup_subs[0], tup_subs[1], x_2)
        return x_2
    else:
        return None

def timestamps_to_dates(x):
    return [dt.datetime.fromtimestamp(t) for t in x]


def reorder_commas(x):
    """
    Function for reordering strings like 'Palmas, Las' into 'Las Palmas'
    """
    l = x.split(',')
    l.reverse()
    l = [s.strip() for s in l]
    return ' '.join(l)

def parse_time_line(x):
    x = x.split(' ')
    n1, n2 = x[:2]
    times = [int(r) for r in x[2:]]
    return n1, n2, times

def subset_edges(net, locs):
    i = 0
    locs = set(locs)
    subs = []
    for edge in net.edges:
        if i in locs:
            subs.append(edge)
        i += 1
    return subs


if __name__=='__main__':
    # Obtain subset from the general file
    # csvgrep -c 15 -m "Galicia" --skip-lines 0 -e "utf-8" -d ";" set2/actives_cityloc_200701_200704.dat > galicia/actives_cityloc_200701_200704.csv
# create subset of canarias from original file
    log_path = '../data/mobile_network/set2/time_res_2007_sorted_sec.mutual_call_unique3.txt'
    subset_path = '../data/mobile_network/madrid/actives_cityloc_200701_200704.csv'
    dic_path = '../data/mobile_network/madrid/madrid.p'
    re_dic_path = '../data/mobile_network/madrid/madrid_reorder.p'
    output_path = '../data/mobile_network/madrid/madrid_call_log.txt'
    create_subset_dictionary(ref_path=subset_path, dic_path=dic_path)
    write_logs_subset(log_path, dic_path, output_path)
    canarias_log_path = '../data/mobile_network/madrid/madrid_call_log.txt'
    canarias_new_id_path = '../data/mobile_network/madrid/madrid_call_newid_log.txt'
    # reorder dic
    reorder_ids(canarias_log_path, re_dic_path)
    reorder_log_ids(canarias_log_path, canarias_new_id_path, re_dic_path)


