import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm 
import gc
import timm
import torch
from torch import nn 
import torch.nn.functional as F 
from sklearn.preprocessing import normalize
from dataset import ShopeeDataset
from augmentations import get_train_transforms, get_valid_transforms

from model import ShopeeModel
from custom_scheduler import ShopeeScheduler

from sklearn.neighbors import NearestNeighbors
from torchmetrics import F1
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import gather_all_tensors

def load_dataloader(CFG):

    df = pd.read_csv(CFG.train_csv)

    tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
    df['target'] = df.label_group.map(tmp)

    valdf = df[df['fold'] == 0].reset_index(drop = True)
    df = df[df['fold'] != 0].reset_index(drop = True)

    labelencoder= LabelEncoder()
    df['label_group'] = labelencoder.fit_transform(df['label_group'])

    labelencoder= LabelEncoder()
    valdf['label_group'] = labelencoder.fit_transform(valdf['label_group'])

    trainset = ShopeeDataset(df,
                             CFG.data_dir,
                             transform = get_train_transforms(img_size = CFG.img_size))

    valset = ShopeeDataset(valdf,
                             CFG.data_dir,
                             transform = get_valid_transforms(img_size = CFG.img_size))

    # tokenizer = AutoTokenizer.from_pretrained('../input/bert-base-uncased')
    # trainset = ShopeeDataset(df, tokenizer = tokenizer)
    # valset = ShopeeDataset(valdf, tokenizer = tokenizer)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = CFG.batch_size,
        num_workers = 8,
        pin_memory = True,
        shuffle = True,
        drop_last = True
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size = CFG.batch_size,
        num_workers = 8,
        pin_memory = True,
        shuffle = False,
        drop_last = False
    )

    return trainloader, valloader

class ShopeeModel_PL(pl.LightningModule):

    def __init__(self, CFG):
        super().__init__()

        self.cfg = CFG
        df = pd.read_csv(CFG.train_csv)
        tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
        df['target'] = df.label_group.map(tmp)
        self.valdf = df[df['fold'] == 0].reset_index(drop = True)
        num_classes = df[df['fold'] != 0]['label_group'].nunique()
        print('Number of classes in training: ', num_classes)
        print('Number of classes in validation: ', self.valdf['label_group'].nunique())

        # self.f1 = F1(num_classes = self.valdf['label_group'].nunique())
        self.model = ShopeeModel(num_classes, CFG.model_name, CFG.fc_dim, CFG.margin, CFG.scale)

        self.scheduler_params = {
        "lr_start": 1e-5,
        "lr_max": 1e-5 * self.cfg.batch_size,     # 1e-5 * 32 (if batch_size(=32) is different then)
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
        }

    def shared_step(self, batch):

        image, label = batch['image'], batch['label']
        _, loss = self.model(image, label)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log_dict({'train_loss': loss})
        return loss

    def validation_step(self,batch, batch_idx):
        features = self.shared_step(batch)

        return features

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, embeds):
        return self.validation_epoch_end(embeds)

    def validation_epoch_end(self, embeds):

        image_embeddings = torch.cat(embeds)  
        image_embeddings = self.sync_across_gpus(image_embeddings)
        # image_embeddings = gather_all_tensors(image_embeddings)
        image_embeddings = image_embeddings.detach().cpu().numpy()
        # assert len(self.valdf) == len(image_embeddings), "They shld match"

        if self.trainer.running_sanity_check:
            # predictions = self.get_image_neighbors(self.valdf, image_embeddings, KNN = 2)
            dummy_valdf = self.valdf.sample(len(image_embeddings))
            final_f1, best_thres = self.get_image_neighbors(dummy_valdf, image_embeddings, KNN = 2)
            # dummy_valdf['oof'] = predictions
            # dummy_valdf['f1'] = dummy_valdf.apply(self.getMetric('oof'),axis=1)
            # final_f1 = dummy_valdf.mean()
            # dummy_valdf = self.valdf.sample(len(image_embeddings))
            # final_f1, best_thres = self.get_cv(dummy_valdf, image_embeddings)
        else:
            # predictions = self.get_image_neighbors(self.valdf, image_embeddings)
            final_f1, best_thres = self.get_image_neighbors(self.valdf, image_embeddings)
            # final_f1, best_thres = self.get_cv(self.valdf, image_embeddings)
            # predictions = self.valdf.groupby('image_phash').posting_id.agg('unique').to_dict()
            # self.valdf['oof'] = self.valdf.image_phash.map(predictions)
            # print(self.valdf['oof'].head())
            # self.valdf['oof'] = predictions
            # self.valdf['f1'] = -1
            # f1 = 0
            # for idx, row in self.valdf.iterrows():
            #     f1_score = self.f1(torch.tensor(row['oof']), torch.tensor(row['target']))
            #     self.valdf.loc[idx, 'f1'] = f1_score
            #     f1 += f1_score

            # self.valdf['f1'] = self.valdf.apply(self.getMetric('oof'),axis=1)
            # final_f1 = self.valdf['f1'].mean()
            # print(f1 / len(self.valdf))
            

        self.log_dict({'val_f1': final_f1, 'best_threshold': best_thres})  
        
        
    def sync_across_gpus(self, t):   # t is a tensor
       
        gather_t_tensor = [torch.zeros_like(t) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_t_tensor, t)
        return torch.cat(gather_t_tensor)

    def configure_optimizers(self):

        # optimizer = torch.optim.Adam(self.model.parameters(),
        #                          lr = self.scheduler_params['lr_start'])
        optimizer = torch.optim.Adam(self.model.parameters(),
                                 lr = self.cfg.lr)
        # scheduler = ShopeeScheduler(optimizer, **self.scheduler_params)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience = 5, factor = 0.5),
                    'monitor': 'val_f1'
                    }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_image_neighbors(self, df, embeddings, KNN=50):

        model = NearestNeighbors(n_neighbors = KNN)
        model.fit(embeddings)
        distances, indices = model.kneighbors(embeddings)
        
        threshold_range = np.arange(0, 7, 0.5)
        results = []
        for threshold in threshold_range:
            predictions = []
            for k in range(embeddings.shape[0]):
                idx = np.where(distances[k,] < threshold)[0]
                ids = indices[k,idx]
                posting_ids = df['posting_id'].iloc[ids].values
                predictions.append(posting_ids)

            df['oof'] = predictions
            df['f1'] = df.apply(self.getMetric('oof'),axis=1)
            final_f1 = df['f1'].mean()

            results.append((final_f1, threshold))

        max_tuple = max(results, key = lambda x: x[0])
        max_score = max_tuple[0]
        max_thres = max_tuple[1]
        print(f'Best score {max_score} obtained for threshold {max_thres}')

        del model, distances, indices
        gc.collect()

        return max_score, max_thres

    def get_cv(self, df, outs):
        thresholds = list(np.arange(0, 0.4, 0.05))
        # thresholds = list(np.arange(0.01, 0.1, 0.01))
        scores = []
        
        # # set target
        # tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
        # df['target'] = df.label_group.map(tmp)

        # Normalize
        outsn = normalize(outs)

        # to torch
        outsn_torch = torch.from_numpy(outsn).cuda()
        
        # calculate cosine simularity with torch cuda()
        distances = 1 - torch.matmul(outsn_torch, outsn_torch.T).cpu().T
        
        for threshold in thresholds:
            predictions = []
            for k in range(outs.shape[0]):
                idx = np.where(distances[k,] < threshold)[0]
                o = df.iloc[idx].posting_id.values
                predictions.append(o)
            df["preds"] = predictions
            #df['oof'] = df.apply(combine_for_cv,axis=1)
            df['f1'] = df.apply(self.getMetric("preds"),axis=1)
            score = df['f1'].mean()
            print(f'Our f1 score for threshold {threshold} is {score}')
            scores.append(score)
        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'Our best score is {best_score} and has a threshold {best_threshold}')
        gc.collect()
        torch.cuda.empty_cache()

        return best_score, best_threshold

    def getMetric(self, col):
        def f1score(row):
            n = len( np.intersect1d(row.target,row[col]) )
            return 2*n / (len(row.target)+len(row[col]))
        return f1score


def load_model(CFG):

    # model = ShopeeModel_PL.load_from_checkpoint(checkpoint_path = './weights/effb4ns_imgsize512_thres45.ckpt', CFG = CFG)
    model = ShopeeModel_PL(CFG)

    return model