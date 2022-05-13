from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import logging
from torch.utils.data import DataLoader
import random
import sys
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import gc

models_name = ['densenet']
feature_extract = False
num_classes = 2
data_dir = "./data"
# data_dir = "/scratch/data/retinal_data"
# model_name = "densenet"
# model_name = "resnet"
# model_name = "vgg"
# model_name = "squeezenet"
# model_name = "alexnet"
# model_name = "inception" batch_size = 7
# model_name = "googlenet"
batch_size = int(sys.argv[1])
num_workers = int(sys.argv[2])
patience = int(sys.argv[3])
project_name = sys.argv[4]
# batch_size = 2
# num_workers = 2
# use_pretrained = False
train = 'train'
val = 'val'
test = 'test'
criterion = CrossEntropyLoss()
api_key = 'wMHJnrgcTvUUwL5cmth3oJrpX'
desired_output = []
actual_output = []


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Classifier(pl.LightningModule):
    def __init__(self, use_pretrained):
        super().__init__()
        model_ft = None
        input_size = 0
        self.prepare_data_per_node = True
        self.use_pretrained = use_pretrained

        if model_name == "googlenet":
            """ GoogLeNet
            """
            model_ft = models.googlenet(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg16(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            # logging.info("Invalid model name, exiting...")
            exit()
        self.model_ft = model_ft
        self.input_size = input_size

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        if model_name == "inception":
            outputs, aux_outputs = self.model_ft(inputs)
            loss1 = self.cross_entropy_loss(outputs, labels)
            loss2 = self.cross_entropy_loss(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
            acc = accuracy(outputs, labels)
        elif model_name == "googlenet" and self.use_pretrained is False:
            outputs = self.model_ft(inputs)
            loss = self.cross_entropy_loss(outputs[0], labels)
            acc = accuracy(outputs[0], labels)
        else:
            outputs = self.model_ft(inputs)
            loss = self.cross_entropy_loss(outputs, labels)
            acc = accuracy(outputs, labels)
        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, acc = self._shared_eval_step(val_batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)
        return metrics

    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        outputs = self.model_ft(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        acc = accuracy(outputs, labels)
        desired_output.append(labels)
        actual_output.append(outputs)
        # loss, acc = self._shared_eval_step(test_batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model_ft(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        acc = accuracy(outputs, labels)
        return loss, acc

    def configure_optimizers(self):
        optimizer_ft = optim.Adam(self.model_ft.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.9)
        return [optimizer_ft], [scheduler]

    def cross_entropy_loss(self, output, labels):
        loss = criterion(output, labels)
        return loss


class RetinalDataModule(pl.LightningDataModule):

    def __init__(self, input_size):
        self.data_loaders_dict = None
        self.input_size = input_size
        self.image_datasets = None
        self.prepare_data_per_node = True
        # self.save_hyperparameters(logger=False)

    def setup(self, stage):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                               [train, val, test]}

    def train_dataloader(self):
        return DataLoader(self.image_datasets[train], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.image_datasets[val], batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.image_datasets[test], batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/metric_1": 0, "hp/metric_2": 0})


def one_hot(lable_lable, pre_pre):
    one_hot_l = []
    one_hot_p = []
    for label in lable_lable:
        for q in label:
            v = [0] * 2
            v[q] = 1
            one_hot_l.append(v)
    for pred in pre_pre:
        for q in pred:
            if q[0] > q[1]:
                p = 0
            else:
                p = 1
            v = [0] * 2
            v[p] = 1
            one_hot_p.append(v)
    return one_hot_l, one_hot_p


if __name__ == '__main__':
    for model_name in models_name:
        use_pretrained = True
        desired_output = []
        actual_output = []
        comet_logger1 = pl_loggers.CometLogger(api_key=api_key,
                                               save_dir="logs/",
                                               project_name=project_name,
                                               experiment_name=f"{model_name}_use_pretrained" if use_pretrained
                                               else f"{model_name}_un-pretrained", )
        model1 = Classifier(use_pretrained)
        data_module = RetinalDataModule(model1.input_size)
        trainer1 = pl.Trainer(logger=comet_logger1,
                              precision=16,
                              gpus=-1,
                              callbacks=[EarlyStopping(monitor="val_loss", patience=patience)],
                              max_epochs=100,
                              min_epochs=3,
                              default_root_dir="./trained_model",
                              check_val_every_n_epoch=1,
                              strategy='dp',
                              )
        trainer1.fit(model1, data_module)
        trainer1.test(model1, data_module)
        trainer1.save_checkpoint(f"trained_model/{model_name}_use_pretrained.ckpt")
        comet_logger1.experiment.log_model(f"{model_name}_use_pretrained.ckpt",
                                           f'trained_model/{model_name}_use_pretrained.ckpt')
        one_hot_labels, one_hot_preds = one_hot(desired_output, actual_output)

        comet_logger1.experiment.log_confusion_matrix(one_hot_labels, one_hot_preds)
        use_pretrained = False
        desired_output = []
        actual_output = []
        comet_logger2 = pl_loggers.CometLogger(api_key=api_key,
                                               save_dir="logs/",
                                               project_name=project_name,
                                               experiment_name=f"{model_name}_use_pretrained" if use_pretrained
                                               else f"{model_name}_no_use_pretrained", )
        model2 = Classifier(use_pretrained)
        data_module = RetinalDataModule(model2.input_size)
        trainer2 = pl.Trainer(logger=comet_logger2,
                              precision=16,
                              gpus=-1,
                              callbacks=[EarlyStopping(monitor="val_loss", patience=patience)],
                              max_epochs=100,
                              min_epochs=3,
                              default_root_dir="./trained_model",
                              check_val_every_n_epoch=1,
                              )
        trainer2.fit(model2, data_module)
        trainer2.test(model2, data_module)
        trainer2.save_checkpoint(f"trained_model/{model_name}_no_use_pretrained.ckpt")
        comet_logger2.experiment.log_model(f"{model_name}_no_use_pretrained.ckpt",
                                           f'trained_model/{model_name}_no_use_pretrained.ckpt')
        one_hot_labels, one_hot_preds = one_hot(desired_output, actual_output)
        comet_logger2.experiment.log_confusion_matrix(one_hot_labels, one_hot_preds)
