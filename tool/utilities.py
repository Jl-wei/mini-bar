import json
import os

from collections import OrderedDict

def read_json(fname):
    with open(fname, 'rt') as f:
        return json.load(f, object_hook=OrderedDict)

def save_json(fname, config):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def df_of_cluster(df, cluster_n):
    return df[df['clusters'] == cluster_n]

def count_cluster_num(df):    
    return df['clusters'].max()+1
