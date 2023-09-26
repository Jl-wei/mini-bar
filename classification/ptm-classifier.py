import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import sys; sys.path.append("..")
from classification.model import AppReviewTagger
from classification.dataset import AppReviewDataset
from tool.utilities import read_json


class Classifier():
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config = read_json(os.path.join(model_path, "config.json"))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        self.model = AppReviewTagger.load_from_checkpoint(
            os.path.join(model_path, "checkpoints/checkpoint.ckpt"),
            config=self.config,
            n_classes=len(self.config["label_columns"])
        ).to(self.device)

    def classify(self, docs):
        if isinstance(docs, pd.core.series.Series):
            docs = docs.to_frame()

        dataset = AppReviewDataset(
            docs,
            self.tokenizer,
            max_token_len=512,
            label_columns=[]
        )

        comment_texts = []
        predictions = []
        
        with torch.no_grad():
            for item in tqdm(dataset):
                _, prediction = self.model(
                    item["input_ids"].unsqueeze(dim=0).to(self.device), 
                    item["attention_mask"].unsqueeze(dim=0).to(self.device)
                )
                comment_texts.append(item["comment_text"])
                predictions.append(prediction.flatten())

        predictions = torch.stack(predictions).detach().cpu().numpy()
        upper, lower = 1, 0
        predictions = np.where(predictions > 0.5, upper, lower)
        predictions = pd.DataFrame(predictions)
        predictions.columns = ["irrelevant", "feature_request", "problem_report"]
        
        df = pd.concat([docs, predictions], axis=1)
        feature_request_df = df[df["feature_request"]==1]
        problem_report_df = df[df["problem_report"]==1]
        irrelevant_df = df[df["irrelevant"]==1]
        
        return feature_request_df, problem_report_df, irrelevant_df


if __name__ == '__main__':
    df = pd.read_csv("../dataset/for_classification/Garmin Connect.en.csv")
    model_path = "../tool/models/xlm-r"
    classifier = Classifier(model_path)
    fr, br, ir = classifier.classify(df['data'])
    
    print(f"Feature Request: \n {fr}")
    print(f"Problem Report: \n {br}")
    print(f"Irrelevant: \n {ir}")
    