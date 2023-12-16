import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


class ProductDataset:
    branches = set()
    mappers = dict()

    def __init__(self, dataframe, archive, n_folds=5, random_seed=42):
        self.archive = archive

        # Load data
        self.df = dataframe

        # Make splits
        self._n_folds = n_folds
        self._random_seed = random_seed
        self._split_folds()

    def _split_folds(self):
        _data = self.df.sample(frac=1, random_state=self._random_seed).fillna('*').reset_index()
        _data['strat_col'] = _data.LVL_1 + ' | ' + _data.LVL_2 + ' | ' + _data.LVL_3 + ' | ' + _data.LVL_4
        kf = StratifiedKFold(n_splits=self._n_folds, shuffle=True, random_state=self._random_seed)

        self.train_idx = [[] for _ in range(self._n_folds)]
        self.val_idx = [[] for _ in range(self._n_folds)]
        self.test_idx = [[] for _ in range(self._n_folds)]
        for i, (all_train, test) in enumerate(kf.split(_data.title, _data.strat_col)):
            train_and_val = _data.iloc[all_train]
            train, val = train_test_split(train_and_val, test_size=0.25, random_state=self._random_seed,
                                          stratify=train_and_val.strat_col)

            self.train_idx[i] = train['index']
            self.val_idx[i] = val['index']
            self.test_idx[i] = _data.iloc[test]['index']

    def from_old_saves(self, archive: str,
                 dataset_path='data/amz_metadata',
                 folds_path='data/folds',
                 n_folds=5, ):

        self.archive = archive

        # Load data
        self.df = pd.read_feather(os.path.join(dataset_path, self.archive + '.fth'))

        # Load saved splits
        self._n_folds = n_folds
        self.train_idx = [np.load(os.path.join(folds_path, str(i), f'{archive}_train_idxs.npy'))
                          for i in range(self._n_folds)]
        self.val_idx = [np.load(os.path.join(folds_path, str(i), f'{archive}_val_idxs.npy'))
                        for i in range(self._n_folds)]
        self.test_idx = [np.load(os.path.join(folds_path, str(i), f'{archive}_test_idxs.npy'))
                         for i in range(self._n_folds)]

    def save_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_split(self, i, fill_unknown_classes=False):
        if fill_unknown_classes:
            train_df = self.df.iloc[self.train_idx[i]].reset_index(drop=True)
            val_df = self.df.iloc[self.val_idx[i]].reset_index(drop=True)
            lvls = [column for column in train_df.keys() if column.startswith('LVL_')]
            for lvl in lvls:
                train_classes = set(train_df[lvl].unique())
                dev_classes = set(val_df[lvl].unique())
                missing_classes = [name for name in dev_classes if name not in train_classes]
                mask = val_df[lvl].isin(missing_classes)
                val_df.loc[mask, lvl] = 'nan'

            return (
                train_df, val_df,
                self.df.iloc[self.test_idx[i]].reset_index(drop=True)
            )

        else:
            return (
                self.df.iloc[self.train_idx[i]].reset_index(drop=True),
                self.df.iloc[self.val_idx[i]].reset_index(drop=True),
                self.df.iloc[self.test_idx[i]].reset_index(drop=True)
            )

    def select_splits(self, i, torch_matrix):
        return (
            torch_matrix[self.train_idx[i].tolist()],
            torch_matrix[self.val_idx[i].tolist()],
            torch_matrix[self.test_idx[i].tolist()]
        )

    def unique_classes(self):
        result = []
        for i in range(100):
            if not f'LVL_{i+1}' in self.df.keys():
                break
            result.append(self.df[f'LVL_{i+1}'].astype(str).unique().shape[0])

        return result
