import torch
import torch.nn as nn

import pytorch_lightning as pl
from torchmetrics.functional import auroc
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel

class AppReviewTagger(pl.LightningModule):
  def __init__(
      self, 
      config,
      n_classes: int, 
      n_training_steps=None, 
      n_warmup_steps=None,
    ):
    super().__init__()
    self.bert = AutoModel.from_pretrained(config['model_name_or_path'])
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCELoss()
    self.label_columns = config["label_columns"]

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  # def on_train_epoch_end(self):
  #   labels = []
  #   predictions = []
  #   for output in self.training_step_outputs:
  #     for out_labels in output["labels"].detach().cpu():
  #       labels.append(out_labels)
  #     for out_predictions in output["predictions"].detach().cpu():
  #       predictions.append(out_predictions)

  #   labels = torch.stack(labels).int()
  #   predictions = torch.stack(predictions)

  #   for i, name in enumerate(self.label_columns):
  #     class_roc_auc = auroc(predictions[:, i], labels[:, i])
  #     self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=2e-5)

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )

    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )