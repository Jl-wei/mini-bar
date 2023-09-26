import pandas as pd
import numpy as np
import os

from datetime import datetime
from sklearn.model_selection import train_test_split

def generate_model_name(config):
    files = config["current_experiment"]['train_and_test_files']
    files = sorted(set(map(lambda f: f.split()[0], files)))

    return f"{config['name']}-{'-'.join(files)}-{datetime.now().isoformat(' ', 'seconds')}"

def get_train_dfs_from_config(config):
    train_df, valid_df, test_df = get_train_valid_test_df_from_config(config)
    if "another_test_df" in config["current_experiment"]:
        another_test_df = get_another_test_df_from_config(config)
    else:
        another_test_df = None
        
    return train_df, valid_df, test_df, another_test_df

def get_train_valid_test_df_from_config(config):
    data_files = get_data_files_from_config(config)
    
    train_df = []
    valid_df = []
    test_df = []
    for data_file in data_files:
        df = pd.read_csv(data_file)
        if config['test_size'] == 1:
            temp_train_df = pd.DataFrame([])
            temp_test_df = df
        else:
            temp_train_df, temp_test_df = train_test_split(
                df, test_size=config['test_size'], 
                stratify = np.array(df[config["label_columns"]])
            )
        if config['valid_size'] > 0:
            temp_train_df, temp_val_df = train_test_split(
                temp_train_df, test_size=config['valid_size']/(1-config['test_size']), 
                stratify = np.array(temp_train_df[config["label_columns"]])
            )
            valid_df.append(temp_val_df)

        train_df.append(temp_train_df)
        test_df.append(temp_test_df)

    train_df = pd.concat(train_df, ignore_index=True)
    if config['valid_size'] > 0: valid_df = pd.concat(valid_df, ignore_index=True)
    test_df = pd.concat(test_df, ignore_index=True)

    return train_df, valid_df, test_df

def get_another_test_df_from_config(config):
    data_files = get_data_files_from_config(config, another=True)
    
    another_test_df = []
    for data_file in data_files:
        df = pd.read_csv(data_file)
        another_test_df.append(df)
        
    another_test_df = pd.concat(another_test_df, ignore_index=True)
    
    return another_test_df

def get_data_files_from_config(config, another=False):
    if another:
        data_files = map(
            lambda file: os.path.join(config['data_folder'], f"{file}.csv"),
            config["current_experiment"]["another_test_df"]
        )
    else:
        data_files = map(
            lambda file: os.path.join(config['data_folder'], f"{file}.csv"),
            config["current_experiment"]["train_and_test_files"]
        )

    return data_files
