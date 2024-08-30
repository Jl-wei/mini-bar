from pathlib import Path
import pandas as pd
import os

import sys; sys.path.append("..")
from tool.utilities import save_json
from hard_clustering import HardClustering


def embed_folder(data_folder, config):
    for data_file in data_folder.iterdir():
        if data_file.is_file() and str(data_file).endswith('.csv'):
            df = pd.read_csv(data_file)
            clustering = HardClustering(config)
            df['embedding'] = clustering.embed(df[config["data_column"]]).tolist()
            df['20_dim_embedding'] = clustering.reduce_dimension(df['embedding'].tolist(), output_dim=20).tolist()
            df['2_dim_embedding'] = clustering.reduce_dimension(df['embedding'].tolist(), output_dim=2).tolist()

            df.to_json('./{}.json'.format(data_file.stem), orient='records')

def makedir(app_name, save_path, config):
    os.makedirs("{}/{}/{}".format(save_path, config["name"], app_name), exist_ok = True)
    os.chdir("{}/{}/{}".format(save_path, config["name"], app_name))
    save_json("{}/{}/config.json".format(save_path, config["name"]), config)

if __name__ == "__main__":
    configs = [
        {
            "name": "multilingual_clip",
            "embedding": "clip-ViT-B-32-multilingual-v1"
        },
        {
            "name": "multilingual_mpnet",
            "embedding": "paraphrase-multilingual-mpnet-base-v2"
        },
        {
            "name": "multilingual_minilm",
            "embedding": "paraphrase-multilingual-MiniLM-L12-v2"
        },
        {
            "name": "multilingual_use",
            "embedding": "distiluse-base-multilingual-cased-v1"
        },
        {
            "name": "xlmr",
            "embedding": "paraphrase-xlm-r-multilingual-v1"
        },
        {
            "name": "labse",
            "embedding": "LaBSE"
        },
        # {
        #     "name": "instructor",
        #     "embedding": "instructor-xl"
        # },
        {
            "name": "multilingual-e5",
            "embedding": "multilingual-e5-large"
        },
        {
            "name": "tfidf",
            "embedding": "tfidf"
        },
        {    
            "name": "bow",
            "embedding": "bow"
        }
    ]

    model = "HDBSCAN"
    data_column = "data"
    data_path = "../dataset/for_clustering/labelled/"
    embedding_path = "../dataset/for_clustering/embedded/"

    for config in configs:
        config['model'] = model
        config['data_column'] = data_column
        path = Path(__file__).parent.joinpath(data_path).resolve()
        save_path = Path(__file__).parent.joinpath(embedding_path).resolve()
        for data_folder in path.iterdir():
            makedir(data_folder.name, save_path, config)
            embed_folder(data_folder, config)
