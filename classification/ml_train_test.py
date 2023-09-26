from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

import os
import random
import logging
import pytorch_lightning as pl
from imp import reload
from pathlib import Path

import utilities
import sys; sys.path.append("..")
from tool.utilities import read_json, save_json

def get_classifier(name):
    if name == 'rf':
        return RandomForestClassifier()
    elif name == 'sgd':
        return SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)
    elif name == 'svm':
        return svm.SVC()
    elif name == 'nb':
        return MultinomialNB()
    else:
        raise('Unknow classifier')

def classify(model_name, target_names, n_epochs, train_df, test_df, another_test_df=None):
    classifier = get_classifier(model_name)
    text_clf = Pipeline([
        ('vect', CountVectorizer(max_df=0.5, min_df=3, stop_words=list(fr_stop)+list(en_stop))),
        ('tfidf', TfidfTransformer()),
        ('mclf', MultiOutputClassifier(classifier))
        ])

    train_data = train_df['data'].values.astype('U')
    train_target = train_df[target_names]
    for i in range(n_epochs):
        text_clf.fit(train_data, train_target)

    evaluate(text_clf, test_df, target_names)

    if another_test_df is not None:
        evaluate(text_clf, another_test_df, target_names)

def evaluate(text_clf, test_df, target_names):
    for lang in test_df['ori_lang'].unique():
        df = test_df.loc[test_df['ori_lang']==lang]
        logging.info(f"App: {test_df['app'].unique()}")
        logging.info(f"Original language: {lang}")
        logging.info(f"Size of test set: {df.shape[0]}")

        test_data = df['data'].values.astype('U')
        test_target = df[target_names]
        predicted = text_clf.predict(test_data)
        report = metrics.classification_report(test_target, predicted, 
                                                target_names=target_names, 
                                                digits=6, zero_division=0)
        logging.info(f"Classification Report: \n {report}")


def main(name, config, train_df, test_df, another_test_df = None):
    reload(logging)
    
    log_path = os.path.join(config["output_dir"], "lightning_logs", name)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(os.path.join(log_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(log_path, "test.csv"), index=False)
    save_json(os.path.join(log_path, "config.json"), config)
    
    logging.basicConfig(
        filename=os.path.join(log_path, "log.log"),
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    classify(config['model_name_or_path'], 
             config['label_columns'], 
             config['n_epochs'],
             train_df, test_df, another_test_df)


if __name__ == '__main__':
    configs = read_json('./ml_config.json')
    
    for config in configs:
        if config['machine_learning'] is not True: continue
        for e in range(len(config["experiments"])):
            for r in range(config["number_runs"]):
                current_config = config.copy()
                current_config['seed'] = random.randrange(0, 10000)
                pl.seed_everything(current_config['seed'])
                current_config["current_experiment"] = config["experiments"][e]
                current_config.pop("experiments", None)

                train_df, _, test_df, another_test_df = utilities.get_train_dfs_from_config(current_config)
                name = utilities.generate_model_name(current_config) + '-' + str(r)
                main(name, current_config, train_df, test_df, another_test_df)
