from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from InstructorEmbedding import INSTRUCTOR

import sklearn.cluster as cls
import numpy as np
import umap.umap_ as umap
import hdbscan
import spacy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class HardClustering:
    def __init__(self, config=None):
        self.config = config
        if config:
            self.embedding = config["embedding"]
            self.model = config["model"]
        else:
            # self.embedding = "instructor"
            self.embedding = "Alibaba-NLP/gte-large-en-v1.5"
            self.model = "HDBSCAN"

    def clustering(self, ds, plot_clusters=False, fig_name='fig', ds_embeded=False):
        if not ds_embeded:
            ds = self.embed(ds)
            ds = self.reduce_dimension(ds)

        clusters = self.clustering_embeddings(ds, self.model)
        if plot_clusters: self.plot_clusters(ds, clusters, fig_name)
        return clusters

    def clustering_embeddings(self, df_embeddings, clustering_model="HDBSCAN"):
        df_embeddings = list(df_embeddings)
        if self.config:
            if clustering_model == 'HDBSCAN':
                clustering_method = hdbscan.HDBSCAN
            else:
                clustering_method = getattr(cls, clustering_model)
            model = clustering_method(**self.config.get('args'))
        else:
            if clustering_model == "KMeans":
                model = cls.KMeans(n_clusters=20)
            elif clustering_model == "DBSCAN":
                model = cls.DBSCAN(eps=0.3)
            elif clustering_model == "OPTICS":
                model = cls.OPTICS(min_samples=5, min_cluster_size=5)
            elif clustering_model == "AgglomerativeClustering":
                model = cls.AgglomerativeClustering(n_clusters=20)
            else:
                model = hdbscan.HDBSCAN(cluster_selection_method ='leaf',
                                        metric='euclidean',
                                        min_cluster_size=5,
                                        min_samples=1)
        return model.fit_predict(df_embeddings)

    def reduce_dimension(self, df_embeddings, output_dim=20):
        reducer = umap.UMAP(n_neighbors=100, n_components=output_dim, 
                            metric='cosine', output_metric='euclidean', min_dist=0)
        return reducer.fit_transform(df_embeddings)

    def embed(self, data_serie):
        if self.embedding in ['en_use_lg', 'xx_use_lg']:
            nlp = spacy.load(self.embedding)
            df_embeddings = np.array(data_serie.apply(lambda x: nlp(str(x)).vector).tolist())
        elif self.embedding == 'bow':
            vectorizer = CountVectorizer(max_df=0.5, min_df=3, stop_words=list(fr_stop)+list(en_stop))
            df_embeddings = vectorizer.fit_transform(data_serie).toarray()
        elif self.embedding == 'tfidf':
            vectorizer = TfidfVectorizer(max_df=0.5, min_df=3, stop_words=list(fr_stop)+list(en_stop))
            df_embeddings = vectorizer.fit_transform(data_serie).toarray()
        # # Remove INSTRUCTOR due to dependency issue
        # # https://github.com/xlang-ai/instructor-embedding/issues/119
        # elif self.embedding.startswith('instructor'):
        #     sentences = list(map(lambda x: ["Represent the app user review for clustering: ", x], data_serie.to_list()))
        #     embedder = INSTRUCTOR('hkunlp/instructor-xl')
        #     df_embeddings = embedder.encode(sentences)
        elif 'e5' in self.embedding:
            embedder = TransformerDocumentEmbeddings('intfloat/multilingual-e5-large')
            df_embeddings = np.array(data_serie.apply(
                    lambda x: embedder.embed(Sentence("query: " + str(x)))[0].embedding.detach().cpu().numpy()
                ).tolist())
        else:
            if "/" in self.embedding:
                embedder = SentenceTransformer(self.embedding)
            else:
                embedder = SentenceTransformer('sentence-transformers/{}'.format(self.embedding))
            df_embeddings = np.array(data_serie.apply(lambda x: embedder.encode(str(x))).tolist())

        return df_embeddings

    def plot_clusters(self, ds, clusters, fig_name, ds_embeded=True):
        ds = list(ds)
        if not ds_embeded: 
            ds = self.embed(ds)
        if len(ds[0]) > 2: 
            ds = self.reduce_dimension(ds, 2)
        ds = np.array(ds)

        df = pd.DataFrame(data = {
            'dim1': ds[:,0],
            'dim2': ds[:,1],
            'clusters': clusters
        })
        outlier_df = df[df['clusters']==-1]
        normal_df = df[df['clusters']!=-1]

        plt.figure()
        plt.scatter(outlier_df['dim1'],
                    outlier_df['dim2'],
                    c='#DCDCDC')
        plt.scatter(normal_df['dim1'],
                    normal_df['dim2'],
                    c=normal_df['clusters'],
                    cmap='rainbow')
        plt.savefig('{}.png'.format(fig_name))
