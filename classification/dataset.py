import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import CamembertTokenizer

class AppReviewDataset(Dataset):

  def __init__(
    self, 
    data: pd.DataFrame, 
    tokenizer: CamembertTokenizer, 
    max_token_len: int = 128,
    label_columns = ["rating", "problem_report", "feature_request", "user_experience"],
    data_column = "data",
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    self.label_columns = label_columns
    self.data_column = data_column
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    comment_text = str(data_row[self.data_column])
    labels = data_row[self.label_columns]

    encoding = self.tokenizer.encode_plus(
      comment_text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      comment_text=comment_text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels.astype(float).values)
    )
