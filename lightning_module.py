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

feature_extract = False
num_classes = 2
data_dir = "./data"
model_name = sys.argv[1]
# model_name = "densenet"
# model_name = "resnet"
# model_name = "vgg"
# model_name = "squeezenet"
# model_name = "alexnet"
# model_name = "inception" batch_size = 7
# model_name = "googlenet"
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
# batch_size = 4
use_pretrained = True
train = 'train'
val = 'val'
criterion = CrossEntropyLoss()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_ft = None
        input_size = 0

        if model_name == "googlenet":
            """ GoogLeNet
            """
            model_ft = models.googlenet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg16(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
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
            logging.info("Invalid model name, exiting...")
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
        else:
            outputs = self.model_ft(inputs)
            loss = self.cross_entropy_loss(outputs, labels)
            acc = accuracy(outputs, labels)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics)
        # self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, acc = self._shared_eval_step(val_batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

        # self.log("val_loss_accuracy", metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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
        }
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                               [train, val]}

    def train_dataloader(self):
        return DataLoader(self.image_datasets[train], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.image_datasets[val], batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True)


if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    comet_logger = pl_loggers.CometLogger(api_key="wMHJnrgcTvUUwL5cmth3oJrpX",
                                          save_dir="logs/",
                                          project_name="default_project",
                                          experiment_name=model_name, )
    # checkpoint_callback = ModelCheckpoint(
    #     filepath='weights.pt',
    #     verbose=True,
    #     monitor='val_loss',
    #     mode='min'
    # )
    torch.cuda.empty_cache()
    model = Classifier()
    data_module = RetinalDataModule(model.input_size)
    trainer = pl.Trainer(precision=16, gpus=-1, callbacks=[EarlyStopping(monitor="val_loss")], max_epochs=100,
                         min_epochs=3,
                         default_root_dir="./trained_model", logger=[tb_logger, comet_logger])
    trainer.fit(model, data_module)
