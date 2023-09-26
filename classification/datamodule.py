from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import AppReviewDataset

class AppReviewDataModule(pl.LightningDataModule):

  def __init__(self, train_df, val_df, test_df,
               tokenizer, batch_size=1, max_token_len=128,
               label_columns=["rating", "problem_report", "feature_request", "user_experience"],
               data_column="data"):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
    self.label_columns = label_columns
    self.data_column = data_column

  def setup(self, stage=None):
    self.train_dataset = AppReviewDataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len,
      label_columns = self.label_columns,
      data_column = self.data_column
    )

    self.val_dataset = AppReviewDataset(
      self.val_df,
      self.tokenizer,
      self.max_token_len,
      label_columns = self.label_columns,
      data_column = self.data_column
    )

    self.test_dataset = AppReviewDataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len,
      label_columns = self.label_columns,
      data_column = self.data_column
    )

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=2
    )

  def val_dataloader(self):
    return DataLoader(
      self.val_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )