from argparse import ArgumentParser
from sbert import BERTDataModule, SBERT
from sentence_transformers import SentenceTransformer, util
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities import seed
import random
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import wandb
from pytorch_lightning.loggers import WandbLogger
import pickle
import time


# from ae.ae import AutoEncoder

def load_dataset(dataset_name='cn', sample=1.0):
    # load dataset
    dataset_path = 'train.csv'

    train_dataset = pd.read_csv(dataset_path, names=['label', 'text'])
    val_dataset = pd.read_csv('val.txt', names=['label', 'text'])
    test_dataset = pd.read_csv('test.txt', names=['label', 'text'])

    train_text = train_dataset.text.values
    val_text = val_dataset.text.values
    test_text = test_dataset.text.values

    # get labels
    train_labels = train_dataset.label.values - 1
    val_labels = val_dataset.label.values - 1
    test_labels = test_dataset.label.values - 1

    return {
        'train': {
            'text': train_text,
            'labels': train_labels
        },
        'val': {
            'text': val_text,
            'labels': val_labels
        },
        'test': {
            'text': test_text,
            'labels': test_labels
        }
    }


def main(hparams):
    seed.seed_everything(1)
    # project_name = hparams.dataset if hparams.project_name == None else hparams.project_name
    filename = 'best'

    print(f'Augmenting: {hparams.augmenting}')

    pretrained_model = "DMetaSoul/sbert-chinese-general-v2"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    datasets = load_dataset(hparams.dataset, hparams.sample)

    print(f"Train data: {len(datasets['train']['text'])}")

    num_labels = len(set(datasets['train']['labels']))

    dm = BERTDataModule(datasets, hparams.batch_size, tokenizer, hparams.max_len)

    model = SBERT(
        pretrained_model=pretrained_model,
        num_labels=num_labels,
        dropout=hparams.dropout,
        lr=hparams.lr,
        aug=hparams.augmenting,
        ae_model=hparams.ae_model,
        ae_hidden=hparams.ae_hidden,
        da_model=hparams.da_model,
        aug_number=hparams.aug_number,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        precision=16,
        max_epochs=hparams.epochs,
        deterministic=True,
        check_val_every_n_epoch=100,
        callbacks=
        [
            EarlyStopping(monitor='val_f1',
                          patience=hparams.patience,
                          mode='max'),
            ModelCheckpoint(
                dirpath='saved/',
                save_weights_only=True,
                monitor='val_mcc',
                mode='max',
                filename=filename
            )
        ]
    )
    trainer.tune(model,dm)
    trainer.fit(model, dm)

    trainer.test(datamodule=dm)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--aug_number', type=float, default=0.2)
    parser.add_argument('--augmenting', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--operator', type=str, default='+')
    parser.add_argument('--dataset', type=str, default='cn')
    parser.add_argument('--sample', type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--ae_model', type=str, default='restful-rain-90')
    parser.add_argument('--da_model', type=str, default='brisk-bush-1')
    parser.add_argument('--ae_hidden', type=int, default=768)
    parser.add_argument('--project_name', type=str, default=None)
    args = parser.parse_args()

    main(args)

# icy-serenity-3
