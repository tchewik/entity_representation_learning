import multiprocessing
import os
import random
import string
import sys
from copy import copy

import fasttext
import fire
import numpy as np
import pandas as pd
from colorama import Fore, Style
from gensim.models.fasttext import load_facebook_model
from nltk.tokenize import wordpunct_tokenize
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from src.hierarchical_classifier import HierarchicalClassifier
from src.product_dataset import ProductDataset


class FasttextClassifier(HierarchicalClassifier):
    def __init__(self, archive: str,
                 n_top_nodes=7,
                 retrain=False,
                 model_dir='models/c_fasttext',
                 pretrained_model_path='models/pretrained_ft/cc.en.300.bin',
                 dataset_path='data/amz_metadata',
                 folds_path='data/folds',
                 n_folds=5,
                 folds=None,
                 max_lvl=5,
                 lr=1.0
                 ):
        """
            :param archive: Name of the corpus part.
            :param n_top_nodes: Number of predicted class labels to consider in the branch construction.
            :param retrain: When set to False (default), only evaluate the previously trained models.
            :param pretrained_model_path: Full path to the `*.bin` pretrained fastText model.
            :param dataset_path: Path to the corpus directory.
            :param folds_path: Path to the directory with information about folds.
            :param n_folds: str  - (optional) Number of folds. Parameter --folds overrides this value if present.
            :param folds: list[int] - (optional) Specific folds. This parameter overrides --n_folds if present.
            :param max_lvl: (optional) Maximum level of hierarchy.
        """
        super().__init__(archive=archive, dataset_path=dataset_path,
                         folds_path=folds_path, n_folds=n_folds, folds=folds, max_lvl=max_lvl)

        self._n_top_nodes = n_top_nodes
        self._retrain = retrain
        self._pretrained_model_path = pretrained_model_path

        self._model_dir = model_dir
        if not os.path.isdir(self._model_dir):
            os.mkdir(self._model_dir)

        self._model = None
        self._lr = lr

    def evaluate(self):

        f1s_lvls = [[] for i in range(self._max_lvl)]
        f1s_avg = []
        accuracy_lvls = [[] for _ in range(self._max_lvl)]

        data = ProductDataset.from_pickle(os.path.join(self._dataset_path, self.archive + '.pkl'))

        folds = self._folds if self._folds else range(self._n_folds)
        for fold in tqdm(folds, desc=Fore.BLUE+f'Processing {self.archive}'+Style.RESET_ALL):
            train, val, test = data.get_split(fold)

            subdir = os.path.join(self._model_dir, f'fold_{fold}')
            model_path = os.path.join(subdir, f'model_q_{self.archive}.fasttext')
            if self._retrain or not os.path.isfile(model_path):
                model = self.fit(train, val, data.mappers)
                if not os.path.isdir(subdir):
                    os.mkdir(subdir)

                model.save_model(model_path)

            else:
                model = self.load_the_save(fold)

            true_lvls = [test[f'LVL_{i}'].map(lambda row: self.decode_class(data.mappers, row, lvl=i))
                         for i in range(1, self._max_lvl + 1)]

            predictions = [model.predict(text, k=self._n_top_nodes)[0]
                           for text in test.title.map(lambda row: ' '.join(wordpunct_tokenize(row.lower())))]

            pred_lvls = [
                [self.all_instances_of_class_from_branch(data.mappers, pred, lvl=i + 1) for pred in predictions]
                for i in range(self._max_lvl)]

            all_labels = [[pred_lvls[i][j] for i in range(self._max_lvl)] for j in range(len(pred_lvls[0]))]
            pred_branches = [self._best_sequence_simple(class_labels, data.branches, data.mappers)
                        for class_labels in all_labels]
            pred_lvls = [[pred[i] for pred in pred_branches] for i in range(self._max_lvl)]

            for i in range(self._max_lvl):
                f1s_lvls[i].append(f1_score(true_lvls[i], pred_lvls[i], average='macro'))

            f1s_avg.append(np.mean([f1s_lvls[i][-1] for i in range(self._max_lvl)]))

            for i in range(1, self._max_lvl + 1):
                true_current_lvl = [' | '.join(map(str, path)) for path in list(zip(*true_lvls[:i]))]
                pred_current_lvl = [' | '.join(pred[:i]) for pred in pred_branches]
                accuracy_lvls[i - 1].append(accuracy_score(true_current_lvl, pred_current_lvl))

        for i in range(self._max_lvl):
            f1 = f1s_lvls[i]
            print(f'F1 (LVL{i + 1}) = {(np.mean(f1) * 100).round(2)} ± {(np.std(f1) * 100).round(2)}')

        print(f'F1 (avg) = {(np.mean(f1s_avg) * 100).round(2)} ± {(np.std(f1s_avg) * 100).round(2)}')

        for i in range(self._max_lvl):
            acc = accuracy_lvls[i]
            print(f'Acc (LVL{i + 1}) = {(np.mean(acc) * 100).round(2)} ± {(np.std(acc) * 100).round(2)}')

    def fit(self, train, val, mappers, max_iter=3):
        def calculate_time(x) -> int:
            P = 0.8
            t_1 = train.shape[0] // 20  # Time for one thread

            s = 1 / ((1 - P) + P / x)
            t = t_1 / s
            return int(t)

        if self._pretrained_model_path[-4:] == '.vec':
            pretrained_vectors = self._pretrained_model_path

        elif self._pretrained_model_path[-4:] == '.bin':
            pretrained_vectors = self._pretrained_model_path[:-4] + '.vec'
            if not os.path.isfile(pretrained_vectors):
                print('You specified path to the *.bin model. Convert to *.vec...\t', end='')
                model = load_facebook_model(self._pretrained_model_path)
                model.wv.save_word2vec_format(pretrained_vectors)
                print('Done!')

        best_f1 = 0
        val_true_lvl1 = [self.decode_class(mappers, line, lvl=1) for line in val.LVL_1]

        train_tmpfile = self._make_tmp_file(train, mappers)
        val_tmpfile = self._make_tmp_file(val, mappers)

        for i in range(max_iter):
            cpus_num = min(multiprocessing.cpu_count(), 15)
            try:
                OPTIONS = {
                    'lr': self._lr,
                    'epoch': 25,
                    'dim': 300,
                    'minCount': 3,
                    'minCountLabel': 3,
                    'minn': 3,
                    'maxn': 10,
                    'neg': 10,
                    'loss': 'ova',
                    'pretrainedVectors': pretrained_vectors,
                    'thread': cpus_num,
                }

                _model = fasttext.train_supervised(input=train_tmpfile,
                                                   # autotuneValidationFile=val_tmpfile,
                                                   # autotuneDuration=calculate_time(cpus_num),
                                                   **OPTIONS
                                                   )

                model_predictions = [_model.predict(text, k=self._n_top_nodes)[0] for idx, text in val.title.items()]
                pred_lvl1 = [self.class_from_branch(mappers, ' '.join(pred), lvl=1) for pred in model_predictions]
                f1 = f1_score(val_true_lvl1, pred_lvl1, average='macro')

                print(f'Retraining trial {i}. Validation F1 = {f1 * 100}')
                if f1 > best_f1:
                    best_f1 = f1
                    model = copy(_model)

            except Exception as e:
                print(e, file=sys.stderr)
                self._lr = max(self._lr - 0.1, 0.01)

        model.quantize(input=train_tmpfile, retrain=True, qnorm=True)

        os.remove(train_tmpfile)
        os.remove(val_tmpfile)
        return model

    def make_lvl_seq(self, row, mapper):
        result = str()
        for i in range(1, self._max_lvl + 1):
            if row[f'LVL_{i}']:
                lvl_label = mapper.get(f'lvl{i}')[0].get(row[f'LVL_{i}'])
                if lvl_label:
                    result += ' __label__' + lvl_label
            else:
                lvl_label = mapper.get(f'lvl{i}')[0].get('nan')
                if lvl_label:
                    result += ' __label__' + lvl_label

        return result

    def _make_tmp_file(self, data, mappers):
        """ Prepares a text file for fasttext, returns path to this file. """

        tempdir = 'temp'
        if not os.path.isdir(tempdir):
            os.mkdir(tempdir)

        ft_labels = data[[f'LVL_{i + 1}' for i in range(self._max_lvl)]].astype(str).apply(
            lambda x: self.make_lvl_seq(x, mappers), axis=1)
        _data = pd.DataFrame({
            'text': data.title.map(lambda row: ' '.join(wordpunct_tokenize(row.lower().strip()))),
            'label': ft_labels
        })

        filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        with open(os.path.join(tempdir, filename), 'w') as f:
            for idx, row in tqdm(_data.iterrows(), desc='Writing fasttext file.'):
                ft_row = f'{row.loc["label"]} {row.loc["text"]}\n'
                f.write(ft_row)

        return os.path.join(tempdir, filename)

    def load_the_save(self, fold):
        return fasttext.load_model(
            os.path.join(self._model_dir, f'fold_{fold}', f'model_q_{self.archive}.fasttext'))


def evaluate(archive, n_folds=5, folds=None, retrain=False, n_top_nodes=7, max_lvl=5, lr=1.0):
    """ This will perform the following steps:
        1) Construct the classifier
        2) Train if:
            a) there are no models under models/c_fasttext/fold_*/{archive}/
            b) flag --retrain is present
        3) Take the trained models, construct final predictions (branches) on test, and evaluate properly.

        :param archive: str  - name of the archive on which you want to evaluate the method
        :param n_folds: int  - number of folds for cross-validation
        :param folds: list[int]  - (Optional) specified folds, overrides --n_folds.
        :param retrain: bool  - whether to retrain the model or not (WARNING: removes the previously trained models)
        :param n_top_nodes: int  - number of top predicted nodes to consider for branch construction
        :param max_lvl: int  - maximum level of hierarchy to consider
        :param lr: float  - learning rate for training the model
    """
    clf = FasttextClassifier(archive, n_top_nodes=n_top_nodes, retrain=retrain, n_folds=n_folds, folds=folds,
                             max_lvl=max_lvl, lr=lr)
    clf.evaluate()


if __name__ == '__main__':
    # python src/classifiers/c_fasttext.py Tools_and_Home_Improvement --retrain
    fire.Fire(evaluate)
