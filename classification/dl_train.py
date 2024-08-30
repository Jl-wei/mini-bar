import torch
import gc
import os
import random
import shutil
from pathlib import Path


import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

import sys; sys.path.append("..")
from datamodule import AppReviewDataModule
from model import AppReviewTagger
from dl_test import test
from tool.utilities import read_json, save_json
import utilities


def train(data_module, train_df, name, config):
    # Setup model
    steps_per_epoch=len(train_df) // config["batch_size"]
    total_training_steps = steps_per_epoch * config["n_epochs"]

    warmup_steps = total_training_steps // 5
    warmup_steps, total_training_steps

    model = AppReviewTagger(
        config,
        n_classes=len(config["label_columns"]),
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
    )

    # Train model
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["log_path"], "checkpoints"),
        filename=name,
        verbose=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    logger = TensorBoardLogger(os.path.join(config["output_dir"]), version=name)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=config["n_epochs"],
        accelerator="gpu"
        # gpus=1,
        # progress_bar_refresh_rate=30
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    trained_model = AppReviewTagger.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        config=config,
        n_classes=len(config["label_columns"])
    )

    return trained_model
    
def main(name, config, train_df, val_df, test_df, another_test_df = None):

    log_path = os.path.join(config["output_dir"], "lightning_logs", name)
    config['log_path'] = log_path
    Path(log_path).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(os.path.join(log_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(log_path, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(log_path, "test.csv"), index=False)
    save_json(os.path.join(log_path, "config.json"), config)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])

    data_module = AppReviewDataModule(
        train_df,
        val_df,
        test_df,
        tokenizer,
        batch_size=config["batch_size"],
        max_token_len=config["max_token_count"],
        label_columns=config["label_columns"],
        data_column=config["data_column"]
    )

    trained_model = train(data_module, train_df, name, config)
    test(trained_model, test_df, tokenizer, config)

    if another_test_df is not None:
        another_test_df.to_csv(os.path.join(log_path, "another_test.csv"), index=False)
        test(trained_model, another_test_df, tokenizer, config)

    # Delete the checkpoints to save disk space
    if not config.get("save_checkpoints"):
        shutil.rmtree(os.path.join(log_path, "checkpoints"))

    del tokenizer
    del data_module
    del trained_model
    gc.collect()

    torch.cuda.empty_cache()

if __name__ == '__main__':
    configs = read_json('./dl_config.json')

    for config in configs:
        if config.get("machine_learning"): continue
        for e in range(len(config["experiments"])):
            for r in range(config["number_runs"]):
                current_config = config.copy()
                current_config['seed'] = random.randrange(0, 10000)
                pl.seed_everything(current_config['seed'])
                current_config["current_experiment"] = config["experiments"][e]
                current_config.pop("experiments", None)

                train_df, val_df, test_df, another_test_df = utilities.get_train_dfs_from_config(current_config)

                if config.get("save_checkpoints"):
                    train_df=val_df=test_df= pd.concat([train_df, val_df, test_df], ignore_index=True)

                name = utilities.generate_model_name(current_config)
                main(name, current_config, train_df, val_df, test_df, another_test_df = another_test_df)
