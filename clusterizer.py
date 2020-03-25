import json
import pandas as pd

"""
Script for clusterizing given the ids
"""

clust_ids_path = '../full_run/reclusters_ids.p', 'r'
week_df_path = '../full_run/week_vec_call.txt'
output_path = '../full_run/clustered_df.txt'



def clusters_to_df(df_path, clusters_path, output_path):
    df = pd.read_csv(df_path, sep=' ')
    clusters = json.load(open(clusters_path, 'r'))

    df_new = df[['0', '1']].copy()

    for k, v in clusters.items():
        df_new['c' + str(k + 1)] = df[v].sum(1)
    df_new.to_csv(output_path, sep=' ', index=False)


clusters_to_df(week_df_path, clust_ids_path, output_path)
