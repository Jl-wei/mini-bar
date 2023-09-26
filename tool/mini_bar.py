import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
import argparse

import sys; sys.path.append("..")
from classification.llm import Classifier
from clustering.hard_clustering import HardClustering
from summarization.summarizer import AbstractiveSummarizer

from utilities import save_json, count_cluster_num

class MiniBAR:
    def __init__(self, classifier, clustering, summarizer):
        self.classifier = classifier
        self.clustering = clustering
        self.summarizer = summarizer

    def analyze(self, df, data_column):
        self.init_report(df)

        fea_df, pro_df, irr_df = self.classify(df[[data_column, 'score', 'thumbsUpCount']])
        self.report['feature_request']['count'] = fea_df.shape[0]
        self.report['problem_report']['count'] = pro_df.shape[0]
        self.report['irrelevant']['count'] = irr_df.shape[0]

        fea_df['clusters'] = self.cluster(fea_df[data_column])
        pro_df['clusters'] = self.cluster(pro_df[data_column])

        print("=====================Summarizing=====================")
        for category, df in [('feature_request', fea_df), ('problem_report', pro_df)]:
            clusters_weight = self.clusters_importance(df)
            for i in tqdm(range(0, count_cluster_num(df))):
                c_dict = {}
                current_df = df[df['clusters'] == i]
                current_ds = current_df[data_column]
                c_dict['summary'] = self.summarize(current_ds)
                c_dict['importance'] = clusters_weight[i]
                c_dict['count'] = current_ds.shape[0]
                c_dict['reviews'] = self.sort_reviews(current_ds, c_dict['summary'])

                self.report[category]['clusters'].append(c_dict)
            self.report[category]['clusters'].sort(key=lambda c:c['importance'], reverse=True)

        save_json(f"./reports/{self.report['app']}.json", self.report)

    def init_report(self, df):
        self.report = {
            'app': df['app'].iloc[0],
            'count': df.shape[0],
            'problem_report': {
                'count': 0,
                'clusters': []
            },
            'feature_request': {
                'count': 0,
                'clusters': []
            },
            'irrelevant': {
                'count': 0,
            }
        }

    def classify(self, ds):
        print("=====================Classifying=====================")
        return self.classifier.classify(ds)

    def cluster(self, ds):
        print("=====================Clustering=====================")
        return self.clustering.clustering(ds)

    def summarize(self, ds):
        return self.summarizer.summarize(ds)

    def clusters_importance(self, df):
        clusters_weight = {}
        for i in tqdm(range(0, count_cluster_num(df))):
            current_df = df[df['clusters'] == i]
            nb_reviews = current_df.shape[0]
            avg_ratings = current_df['score'].mean()

            nb_thumbs = current_df['thumbsUpCount'].sum() * 0.1
            clusters_weight[i] = (nb_reviews + nb_thumbs)/avg_ratings
            
        return clusters_weight
    
    def sort_reviews(self, ds, summary):
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

        reviews_embeddings = model.encode(ds.values.astype("str"), convert_to_tensor=True)
        summary_embeddings = model.encode([summary], convert_to_tensor=True)
        cosine_scores = util.cos_sim(reviews_embeddings, summary_embeddings)

        pairs = pd.DataFrame(data= {'review': ds, 
                                    'score': cosine_scores.cpu().squeeze().tolist()})
        pairs = pairs.sort_values(by='score', ascending=False)
        
        return pairs['review'].tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mini-BAR')
    parser.add_argument('--file', action='store')
    args = parser.parse_args()

    mini_bar = MiniBAR(
        classifier=Classifier(),
        clustering=HardClustering(),
        summarizer=AbstractiveSummarizer(),
    )
    
    df = pd.read_csv(args.file)
    mini_bar.analyze(df, 'data')
