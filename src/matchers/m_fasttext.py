import os

import fasttext
import fire
import numpy as np
import pandas as pd
import torch
from colorama import Fore, Style
from nltk.tokenize import wordpunct_tokenize
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.extmath import weighted_mode
from torch.nn.functional import normalize
from tqdm import tqdm

from src.hierarchical_classifier import HierarchicalClassifier
from src.product_dataset import ProductDataset


def process_chunk(chunk: torch.Tensor, known_embeddings: torch.Tensor, known_labels: pd.Series, n_neighbors: int, cuda_device):
    # Compute embeddings and get labels for a chunk of texts
    results = []  # found labels
    for emb in chunk:
        sim = emb.to(cuda_device) @ known_embeddings.to(cuda_device).T
        if n_neighbors == 1:
            results.append(known_labels.iloc[torch.argmax(sim).cpu().item()])
        else:
            dists, indices = torch.topk(sim, k=n_neighbors)
            w_mode = weighted_mode(known_labels.iloc[indices.cpu().numpy()].tolist(),
                                   dists.cpu().numpy())[0][0]
            results.append(w_mode)

    return results


class KNNFasttextMatcher(HierarchicalClassifier):
    def __init__(self, archive: str,
                 n_neighbors=1,
                 dataset_path='data/amz_metadata',
                 folds_path='data/folds',
                 n_folds=5,
                 folds=None,
                 max_lvl=5,
                 model_path='models/pretrained_ft/cc.en.300.bin',
                 cuda_device=-1):
        """
            :param archive: Name of the corpus part.
            :param n_neighbors: int  - Number of neighbors for weighted kNN.
            :param dataset_path: Path to the corpus directory.
            :param folds_path: Path to the directory with information about folds.
            :param n_folds: str  - (optional) Number of folds. Parameter --folds overrides this value if present.
            :param folds: list[int] - (optional) Specific folds. This parameter overrides --n_folds if present.
            :param max_lvl: int  - (optional) Maximum level of hierarchy.
            :param model_path: str  - Full path to the pretrained transformer.
            :param cuda_device: int  - Number of the cuda device for the kNN.
        """
        super().__init__(archive=archive, dataset_path=dataset_path,
                         folds_path=folds_path, n_folds=n_folds, folds=folds, max_lvl=max_lvl)

        self._model = fasttext.load_model(model_path)
        self._n_neighbors = n_neighbors

        cuda = 'cpu' if cuda_device == -1 else f'cuda:{cuda_device}'
        self._cuda_device = torch.device(cuda)

    def _get_tokens(self, data):
        return data.title.map(lambda row: ' '.join(wordpunct_tokenize(row.lower())))

    def fit(self, data):
        tokens = self._get_tokens(data)
        X = np.vstack(tokens.map(self._model.get_sentence_vector))
        y = data['LVL_1'].astype(str)
        for i in range(2, self._max_lvl + 1):
            y += ' | ' + data[f'LVL_{i}'].astype(str)

        self._known_embeddings = normalize(torch.from_numpy(X)).half().to(self._cuda_device)
        self._known_labels = y

    def predict(self, data):
        tokens = self._get_tokens(data)
        X = torch.from_numpy(np.vstack(tokens.map(self._model.get_sentence_vector))).half().to(self._cuda_device)
        return self.unsupervised_match(X)

    def unsupervised_match(self, text_emb: torch.Tensor):
        """
        :param text_emb: Embeddings of the input texts
        :return: Label derived with kNN
        """

        results = []  # found labels
        text_emb = normalize(text_emb)
        for emb in text_emb:
            sim = emb @ self._known_embeddings.T
            if self._n_neighbors == 1:
                results.append(self._known_labels.iloc[torch.argmax(sim).cpu().item()])
            else:
                dists, indices = torch.topk(sim, k=self._n_neighbors)
                w_mode = weighted_mode(self._known_labels.iloc[indices.cpu().numpy()].tolist(),
                                       dists.cpu().numpy())[0][0]
                results.append(w_mode)

        return results

    def evaluate(self):
        all_f1s_lvls = [[] for i in range(self._max_lvl)]
        f1s_avg = []
        accuracy_lvls = [[] for _ in range(self._max_lvl)]

        data = ProductDataset.from_pickle(os.path.join(self._dataset_path, self.archive + '.pkl'))

        folds = self._folds if self._folds else range(self._n_folds)
        for fold in tqdm(folds, desc=Fore.BLUE + f'Processing {self.archive}' + Style.RESET_ALL):
            train, val, test = data.get_split(fold)

            self.fit(train)
            predictions = self.predict(test)

            if type(predictions) == pd.Series:
                pred_branches = predictions.map(lambda row: row.split(' | '))
            else:
                pred_branches = [row.split(' | ') for row in predictions]

            true_lvls = [test[f'LVL_{i}'].tolist() for i in range(1, self._max_lvl + 1)]
            f1_lvls = []
            for i in range(self._max_lvl):
                if type(pred_branches) == pd.Series:
                    pred = pred_branches.map(lambda row: row[i]).astype(str)
                else:
                    pred = [branch[i] for branch in pred_branches]
                f1_lvls.append(f1_score(true_lvls[i], pred, average='macro'))

            for i in range(len(f1_lvls)):
                all_f1s_lvls[i].append(f1_lvls[i])

            f1s_avg.append(np.mean([all_f1s_lvl[-1] for all_f1s_lvl in all_f1s_lvls]))

            for i in range(1, self._max_lvl + 1):
                true_current_lvl = [' | '.join(map(str, path)) for path in list(zip(*true_lvls[:i]))]
                pred_current_lvl = [' | '.join(pred[:i]) for pred in pred_branches]
                accuracy_lvls[i - 1].append(accuracy_score(true_current_lvl, pred_current_lvl))

        for i in range(self._max_lvl):
            f1 = all_f1s_lvls[i]
            print(f'F1 (LVL{i + 1}) = {(np.mean(f1) * 100).round(2)} ± {(np.std(f1) * 100).round(2)}')

        print(f'F1 (avg) = {(np.mean(f1s_avg) * 100).round(2)} ± {(np.std(f1s_avg) * 100).round(2)}')

        for i in range(self._max_lvl):
            acc = accuracy_lvls[i]
            print(f'Acc (LVL{i + 1}) = {(np.mean(acc) * 100).round(2)} ± {(np.std(acc) * 100).round(2)}')


def evaluate(archive, n_neighbors, n_folds=5, folds=None, cuda_device=0, max_lvl=5):
    """ This will perform the following steps:
        1) Construct the weighted kNN classifier
        2) Predict the branches on test, and evaluate properly.

        :param archive: str  - name of the archive on which you want to evaluate the method
        :param n_folds: int  - number of folds for cross-validation
        :param folds: list[int]  - (Optional) specified folds, overrides --n_folds.
        :param n_neighbors: int  - number of neighbors for kNN
        :param cuda_device: int  - cuda device for kNN (one per task)
        :param max_lvl: int  - maximum level of hierarchy to consider
    """
    matcher = KNNFasttextMatcher(archive, n_neighbors=n_neighbors, cuda_device=cuda_device,
                                 n_folds=n_folds, folds=folds, max_lvl=max_lvl)
    matcher.evaluate()


if __name__ == '__main__':
    # python src/matchers/m_fasttext.py Office_Products --n_neighbors 1 --n_folds 5 --cuda_device 0
    fire.Fire(evaluate)
