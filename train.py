import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import timm
import torch
from torch import nn 
import torch.nn.functional as F 

import argparse
import engine
from dataset import ShopeeDataset
from model import ShopeeModel
from custom_scheduler import ShopeeScheduler
from augmentations import get_train_transforms

from utils import load_dataloader, load_model
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

class CFG:
    seed = 54
    img_size = 256
    classes = 11014
    scale = 30
    margin = 0.5
    fc_dim = 512
    epochs = 15
    batch_size = 8
    num_workers = 8
    model_name = 'tf_efficientnet_b1'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler_params = {
        "lr_start": 1e-5,
        "lr_max": 1e-5 * batch_size,     # 1e-5 * 32 (if batch_size(=32) is different then)
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }




def run_training(CFG):
 
    df = pd.read_csv(CFG.train_csv)

    labelencoder= LabelEncoder()
    df['label_group'] = labelencoder.fit_transform(df['label_group'])

    trainset = ShopeeDataset(df,
                             CFG.data_dir,
                             transform = get_train_transforms(img_size = CFG.img_size))

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = CFG.batch_size,
        num_workers = CFG.num_workers,
        pin_memory = True,
        shuffle = True,
        drop_last = True
    )

    model = ShopeeModel(CFG.classes, CFG.model_name, CFG.fc_dim, CFG.margin, CFG.scale)
    model.to(CFG.device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = CFG.scheduler_params['lr_start'])
    scheduler = ShopeeScheduler(optimizer, **CFG.scheduler_params)

    for epoch in range(CFG.epochs):
        avg_loss_train = engine.train_fn(model, trainloader, optimizer, scheduler, epoch, CFG.device)
        torch.save(model.state_dict(), MODEL_PATH + 'arcface_512x512_{}.pt'.format(CFG.model_name))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            },
            MODEL_PATH + 'arcface_512x512_{}_checkpoints.pt'.format(CFG.model_name)
        )

# Ravindra Naik
# Balashivanathan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required = True)
    parser.add_argument('--data_dir', type=str, default='/SSD1TB/Shopee/train_images')
    parser.add_argument('--train_csv', type=str, default='/SSD1TB/Shopee/baseline/folds.csv')
    parser.add_argument('--model_path', type=str, default='./')
    parser.add_argument('--model_name', type=str, default='tf_efficientnet_b5')
    parser.add_argument('--epochs', type=int, default=30)
    # parser.add_argument('--classes', type=int, default=11014)
    parser.add_argument('--scale', type=int, default=30)
    parser.add_argument('--fc_dim', type=int, default=512)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=3e-4)

    args = parser.parse_args()

    trainloader, valloader = load_dataloader(args)
    model = load_model(args)

    ckpt_logger = ModelCheckpoint(dirpath = './weights', filename = args.run_name, verbose = True, \
        save_weights_only= True, monitor = 'val_f1', mode = 'max')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandblogger = WandbLogger(name = args.run_name, project = 'shopee')

    trainer = pl.Trainer(gpus = [args.gpu], max_epochs = args.epochs, accelerator='ddp', \
         plugins=DDPPlugin(find_unused_parameters=False), callbacks = [ckpt_logger, lr_monitor], logger = wandblogger,
         precision = 16)
    trainer.fit(model, trainloader, valloader)
    # trainer.test(model, valloader)

# Best score 0.6091370579749698 obtained for threshold 4.0