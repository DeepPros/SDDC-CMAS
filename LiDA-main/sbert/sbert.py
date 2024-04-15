from sentence_transformers import SentenceTransformer, util

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.utilities import seed
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# import torchmetrics
# from torchmetrics import F1

# import warnings
# warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup,
)

from ae import AutoEncoder


# from ae import autoencoder
# /home/NCKU/2021/Research/LiDA/ae/ae.py


class BERTDataset:
    def __init__(self, text, label, tokenizer, max_len):
        self.text = text
        self.target = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.float),
        }


class BERTDataModule(pl.LightningDataModule):
    def __init__(self, datasets, batch_size, tokenizer, max_len):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = BERTDataset(
            text=datasets['train']['text'],
            label=datasets['train']['labels'],
            tokenizer=tokenizer,
            max_len=max_len
        )

        self.valid_dataset = BERTDataset(
            text=datasets['val']['text'],
            label=datasets['val']['labels'],
            tokenizer=tokenizer,
            max_len=max_len
        )

        self.test_dataset = BERTDataset(
            text=datasets['test']['text'],
            label=datasets['test']['labels'],
            tokenizer=tokenizer,
            max_len=max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )


class attention1d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention1d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, ratio=0.25, stride=1, padding=1, dilation=1, groups=1,
                 bias=True, K=4, temperature=34, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention1d(in_planes, ratio, K, temperature).to('cuda')
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, y):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(y)

        batch_size, in_planes, height = x.size()
        x = x.view(1, -1, height, )  # 变化成一个维度进行组卷积

        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes,
                                                                    self.in_planes // self.groups, self.kernel_size, )
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias.to('cuda')).view(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-1))

        return output


class SBERT(pl.LightningModule):
    def __init__(self, pretrained_model=None, num_labels=None, dropout=None, lr=None, aug=False, ae_model=None,
                 ae_hidden=768, da_model=None, aug_number=0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model, output_hidden_states=True)
        self.bert_drop = nn.Dropout(dropout)
        self.out = nn.Linear(768, num_labels)  # original 768 OR cat 1536
        self.lr = lr
        self.num_labels = num_labels
        # elf.aug = aug
        self.aug = False
        self.ae_model = ae_model
        self.ae_hidden = ae_hidden
        self.da_model = da_model
        self.aug_number = aug_number
        self.DynamicConv = Dynamic_conv1d(in_planes=1, out_planes=1)
        self.save_hyperparameters()
        self.target = []
        self.true = []

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, ids, mask, train=True):
        output = self.bert(
            ids,
            attention_mask=mask,
        )

        pooler = self.mean_pooling(output, mask)

        if self.aug:
            if train:
                ae = AutoEncoder.load_from_checkpoint(
                    f'/root/autodl-tmp/LiDA/ae/best/ae-quora-den-{self.ae_model}.ckpt', embedding_dim=768,
                    hidden_dim=self.ae_hidden, lr=1e-4
                ).cuda()
                da = AutoEncoder.load_from_checkpoint(
                    f'/root/autodl-tmp/LiDA/ae/best/ae-quora-den-{self.da_model}.ckpt', embedding_dim=768,
                    hidden_dim=self.ae_hidden, lr=1e-4
                ).cuda()

                train_aug_lin = (pooler + self.aug_number).to('cuda')
                train_aug_ae = ae(pooler).to('cuda').detach()
                train_aug_da = da(pooler).to('cuda').detach()

                pooler = torch.cat((pooler, train_aug_lin, train_aug_ae, train_aug_da), 0)

                conv_aug_lin = self.DynamicConv(train_aug_lin.unsqueeze(1),torch.cat((train_aug_ae, train_aug_da),dim = 1).unsqueeze(1))
                conv_aug_da = self.DynamicConv(train_aug_da.unsqueeze(1),torch.cat((train_aug_ae, train_aug_lin),dim = 1).unsqueeze(1))
                conv_aug_ae = self.DynamicConv(train_aug_ae.unsqueeze(1),torch.cat((train_aug_da, train_aug_lin),dim = 1).unsqueeze(1))

                pooler = torch.cat((pooler, conv_aug_lin.squeeze(1), conv_aug_ae.squeeze(1), conv_aug_da.squeeze(1)), 0)

        bo = self.bert_drop(pooler)
        output = self.out(bo)

        return output


    def training_step(self, batch, batch_idx):

        ids = batch["ids"].long()
        mask = batch["mask"].long()
        targets = batch["targets"].long()

        if self.aug:
            targets = torch.cat((targets, targets, targets, targets), 0)

        outputs = self(
            ids=ids,
            mask=mask,
            train=True
        )
        loss = F.cross_entropy(outputs, targets)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ids = batch["ids"]
        mask = batch["mask"]
        targets = batch["targets"].long()
        outputs = self(
            ids=ids,
            mask=mask,
            train=False
        )
        loss = F.cross_entropy(outputs, targets)
        outputs = torch.argmax(outputs, dim=-1).float()
        result = {'y_true': targets, 'y_pred': outputs}
        #         print(result)
        self.log('val_loss', loss)
        return result

    def validation_epoch_end(self, output):
        y_true = []
        y_pred = []
        for out in output:
            y_true.extend(out['y_true'].long().cpu().detach().numpy().tolist())
            y_pred.extend(out['y_pred'].long().cpu().detach().numpy().tolist())
        val_acc = metrics.accuracy_score(y_true, y_pred)
        val_f1 = metrics.f1_score(
            y_true, y_pred, average='macro' if self.num_labels > 2 else 'binary'
        )
        val_mcc = metrics.matthews_corrcoef(y_true, y_pred)
        self.log('val_acc', val_acc)
        self.log('val_f1', val_f1)
        self.log('val_mcc', val_mcc)
        return val_f1

    def test_step(self, batch, batch_idx):
        ids = batch["ids"]
        mask = batch["mask"]
        targets = batch["targets"]
        outputs = self(
            ids=ids,
            mask=mask,
            train=False
        )
        outputs = torch.argmax(outputs, dim=-1).float()
        result = {'y_true': targets, 'y_pred': outputs}
        self.true.extend([int(x) for x in result['y_true'].cpu().tolist()])
        self.target.extend([int(x) for x in result['y_pred'].cpu().tolist()])

        #         self.log('val_loss', loss)
        return result

    def test_epoch_end(self, output):
        y_true = []
        y_pred = []
        for out in output:
            y_true.extend(out['y_true'].long().cpu().detach().numpy().tolist())
            y_pred.extend(out['y_pred'].long().cpu().detach().numpy().tolist())
        test_acc = metrics.accuracy_score(y_true, y_pred)
        test_f1 = metrics.f1_score(
            y_true, y_pred, average='macro'
            #             y_true, y_pred, average='macro' if self.num_labels > 2 else 'binary'
        )
        test_mcc = metrics.matthews_corrcoef(y_true, y_pred)
        self.log('test_f1', test_f1)
        self.log('test_acc', test_acc)
        self.log('test_mcc', test_mcc)

        cm = confusion_matrix(self.true, self.target)
        print(cm)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #         return optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 40
            }
        }