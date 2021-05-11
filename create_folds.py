import pandas as pd 
from sklearn.model_selection import GroupKFold

data = pd.read_csv('../train.csv')
data['posting_id'] = data['posting_id'].apply(lambda x: int(x.split('_')[-1]))
gkf = GroupKFold(n_splits = 5)

data['fold'] = -1
for idx, (train_idx, val_idx) in enumerate(gkf.split(data, data['image_phash'], data['label_group'])):
    data.loc[val_idx, 'fold'] = idx

tmp = data.groupby('label_group').posting_id.agg('unique').to_dict()
data['target'] = data.label_group.map(tmp)

print(data.head())

data.to_csv('./folds.csv', index = False)