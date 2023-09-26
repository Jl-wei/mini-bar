import numpy as np
import torch
import logging

from tqdm.auto import tqdm
from sklearn.metrics import classification_report

from dataset import AppReviewDataset

import sys; sys.path.append("..")

def test(trained_model, test_df, tokenizer, config):
    for lang in test_df['ori_lang'].unique():
        df = test_df.loc[test_df['ori_lang']==lang]
        logging.info(f"App: {test_df['app'].unique()}")
        logging.info(f"Original language: {lang}")
        logging.info(f"Size of test set: {df.shape[0]}")
        test_one_lang(trained_model, df, tokenizer, config)

def test_one_lang(trained_model, test_df, tokenizer, config):
    trained_model.eval()
    trained_model.freeze()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = trained_model.to(device)

    test_dataset = AppReviewDataset(
        test_df,
        tokenizer,
        max_token_len=config["max_token_count"],
        label_columns=config["label_columns"]
    )

    comment_texts = []
    predictions = []
    labels = []

    for item in tqdm(test_dataset):
        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0).to(device), 
            item["attention_mask"].unsqueeze(dim=0).to(device)
        )
        comment_texts.append(item["comment_text"])
        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())

    predictions = torch.stack(predictions).detach().cpu()
    labels = torch.stack(labels).detach().cpu()

    y_pred = predictions.numpy()
    upper, lower = 1, 0
    y_pred = np.where(y_pred > config["threshold"], upper, lower)

    y_true = labels.numpy()

    report = classification_report(
        y_true, 
        y_pred, 
        target_names=config["label_columns"], 
        digits=6,
        zero_division=0
    )

    logging.info(f"Classification Report: \n {report}")
