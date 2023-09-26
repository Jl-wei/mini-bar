import pandas as pd
import json
import argparse

from sklearn import metrics
from pathlib import Path


# python evaluate.py --name "multi-hdbscan" --length 1 --scale 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', action='store')
    parser.add_argument('--length', action='store', type=int)
    parser.add_argument('--scale', action='store', type=int)
    args = parser.parse_args()
    
    folder_name = args.name
    length = args.length
    scale = args.scale
    
    path = Path(__file__).parent.joinpath("./output/{}".format(folder_name)).resolve()
    results = {}
    for model_folder in path.iterdir():
        score_dict = {
            "pro": 0,
            "fea": 0
        }
        
        for app_folder in model_folder.iterdir():
            app_name = app_folder.name
            if not app_folder.is_dir(): continue
            for data_file in app_folder.iterdir():
                if data_file.suffix != '.csv' : continue
                df = pd.read_csv(data_file)

                # Don't compare with the clusters with size < 5
                gt = df.pivot_table(index = ['ground_truth'], aggfunc ='size')
                df = df[df['ground_truth'].isin(gt[gt>=5].index.tolist())]

                category_name = data_file.stem[:3]

                labels_true = df['ground_truth']
                labels_pred = df['clusters']

                score = metrics.normalized_mutual_info_score(labels_true, labels_pred)
                # score = metrics.adjusted_rand_score(labels_true, labels_pred)

                score_dict[category_name] += score

        print(model_folder.name)
        for key, value in score_dict.items():
            print(f"{key.upper()}: {value/3}")
        print('***********************')
        value = score_dict["pro"] + score_dict["fea"]
        print(f"Average: {value/6}")

        print('**************************************************************')
        
        embedding_method = "-".join(model_folder.name.split('-')[1:-1])
        if not embedding_method in results: 
            results[embedding_method] = {}
            for cat in ['pro','fea','avg']:
                results[embedding_method][cat] = list(range(length))
        epsilon = float(model_folder.name.split('-')[-1])
        index = round(epsilon * scale)
        for lang in ['bi']:
            results[embedding_method]["pro"][index] = score_dict["pro"] / 3
            results[embedding_method]["fea"][index] = score_dict["fea"] / 3
            results[embedding_method]["avg"][index] = (score_dict["pro"]+ score_dict["fea"])/6

    print(json.dumps(results, indent = 4), sep = "\n")
    for model, model_val in results.items():
        print('**************************************************************')
        print(model)
        for lang, lang_val in model_val.items():
            print(lang)
            i = 0
            for val in lang_val:
                print(round(i, 2), ' ', round(val, 4), '\\\\')
                i += 1/scale
