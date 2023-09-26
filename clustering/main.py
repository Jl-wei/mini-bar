from pathlib import Path
import pandas as pd
import os

import sys; sys.path.append("..")
from tool.utilities import read_json, save_json
from hard_clustering import HardClustering


def clustering_folder(data_folder, experiment_name, config):
    makedir(experiment_name, data_folder.name, config)
    for data_file in data_folder.iterdir():
        if data_file.is_file() and data_file.suffix==".json":
            df = pd.read_json(data_file, orient='records')
            if config['lang'] in ['en', 'fr']:
                df = df[df['ori_lang']==config['lang']]

            clustering = HardClustering(config)
            df['clusters'] = clustering.clustering(df[config["data_column"]], ds_embeded=True)
            clustering.plot_clusters(df[config["plot_data_column"]], df['clusters'], data_file.stem)

            df = df[['data','ground_truth','clusters','ori_lang','app']]
            df.to_csv('./{}.csv'.format(data_file.stem), index=False)

def makedir(experiment_name, app_name, config):
    dir = os.path.dirname(os.path.realpath(__file__))
    os.makedirs("{}/output/{}/{}/{}".format(dir, experiment_name, config['name'], app_name), exist_ok = True)
    os.chdir("{}/output/{}/{}/{}".format(dir, experiment_name, config['name'], app_name))
    save_json("{}/output/{}/{}/config.json".format(dir, experiment_name, config['name']), config)

if __name__ == "__main__":
    configs = read_json('./config.json')

    experiment_name = configs['experiment_name']
    for config in configs['experiments']:
        config['experiment_name'] = configs['experiment_name']
        config['data_column'] = configs['data_column']
        config['plot_data_column']= configs['plot_data_column']
        config['model'] = configs['model']
        config['args'] = configs['args']
        config['lang'] = configs['lang']

        name = config['name']
        if config["model"] == "HDBSCAN":
            # for i in range(0, 11):
            i = 0
            config['args']['cluster_selection_epsilon'] = i / 10
            config['name'] = "{}-{}".format(name, config['args']['cluster_selection_epsilon'])

            path = Path(__file__).parent.joinpath(config["data_dir"]).resolve()
            for data_folder in path.iterdir():
                if data_folder.is_dir():
                    clustering_folder(data_folder, experiment_name, config)
        else:
            raise('Should use HDBSCAN')
