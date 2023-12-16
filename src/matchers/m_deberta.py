import os

import fire
import numpy as np
import torch
from colorama import Fore, Style
from nltk.tokenize import wordpunct_tokenize
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline

from src.hierarchical_classifier import HierarchicalClassifier
from src.product_dataset import ProductDataset

tqdm.pandas()


class KNNDeBERTaMatcher(HierarchicalClassifier):
    def __init__(self, archive: str,
                 retrain=False,
                 n_neighbors=1,
                 dataset_path='data/amz_metadata',
                 folds_path='data/folds',
                 n_folds=5,
                 folds=None,
                 max_lvl=5,
                 model_name='microsoft/deberta-v3-base',
                 cuda_device=-1):
        """
            :param archive: Name of the corpus part.
            :param retrain: bool  - When set to False (default), only evaluate the previously predicted embeddings.
            :param n_neighbors: int  - Number of neighbors for weighted kNN.
            :param dataset_path: Path to the corpus directory.
            :param folds_path: Path to the directory with information about folds.
            :param n_folds: str  - (optional) Number of folds. Parameter --folds overrides this value if present.
            :param folds: list[int] - (optional) Specific folds. This parameter overrides --n_folds if present.
            :param max_lvl: int  - (optional) Maximum level of hierarchy.
            :param model_name: str  - Name on Huggingface or full path to the pretrained transformer.
            :param cuda_device: int  - Number of the cuda device for the embedding prediction (if --retrain) and kNN.
        """
        super().__init__(archive=archive, dataset_path=dataset_path,
                         folds_path=folds_path, n_folds=n_folds, folds=folds, max_lvl=max_lvl)

        self._n_neighbors = n_neighbors
        self._retrain = retrain
        self._model_name = model_name

        self._cuda_int = cuda_device
        cuda = 'cpu' if cuda_device == -1 else f'cuda:{cuda_device}'
        self._cuda_device = torch.device(cuda)

        if self._n_neighbors > 1:
            self._knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1, weights='distance')

    @staticmethod
    def _get_tokens(data):
        return data.title.map(lambda row: ' '.join(wordpunct_tokenize(row.lower())))

    def fit(self, train_df, train_embeddings):
        y = train_df['LVL_1'].astype(str)
        for i in range(2, self._max_lvl + 1):
            y += ' | ' + train_df[f'LVL_{i}'].astype(str)

        if self._n_neighbors == 1:
            print(f'{type(train_embeddings) =}')
            self._known_embeddings = normalize(train_embeddings).half().to(self._cuda_device)
            self._known_labels = y

        else:
            self._knn.fit(train_embeddings.numpy().astype('float32'), y)

    def predict(self, text_emb):
        if self._n_neighbors == 1:
            results = []
            text_emb = normalize(text_emb).half().to(self._cuda_device)
            for emb in text_emb:
                sim = emb @ self._known_embeddings.T
                results.append(torch.argmax(sim).cpu().item())
            return self._known_labels.iloc[results].reset_index(drop=True)
        else:
            X = text_emb.numpy().astype('float32')
            return self._knn.predict(X)

    def collect_embeddings(self, data):
        embedding_pipeline = pipeline(task='feature-extraction',
                                      model=self._model_name,
                                      tokenizer=self._model_name,
                                      device=self._cuda_int)

        # DataLoader handles batching
        loader = DataLoader(data.title.tolist(), batch_size=1024)

        # Generate embeddings in batches
        all_embeddings = []
        for batch in tqdm(loader, desc=Fore.BLUE+f'Getting embeddings for {self.archive}'+Style.RESET_ALL):
            embeddings = embedding_pipeline(batch)
            embeddings = [torch.mean(torch.Tensor(emb[0]), dim=0) for emb in embeddings]
            all_embeddings += embeddings

        # Concatenate batch results
        embeddings = torch.vstack(all_embeddings)
        torch.save(embeddings, os.path.join(self._dataset_path, self.archive + '_emb.pt'))

        return embeddings

    def evaluate(self):

        data = ProductDataset.from_pickle(os.path.join(self._dataset_path, self.archive + '.pkl'))

        embeddings_path = os.path.join(self._dataset_path, self.archive + '_emb.pt')
        if not os.path.isfile(embeddings_path) or self._retrain:
            embeddings = self.collect_embeddings(data.df)
        else:
            embeddings = torch.load(embeddings_path)

        all_f1s_lvls = [[] for i in range(self._max_lvl)]
        f1s_avg = []
        accuracy_lvls = [[] for _ in range(self._max_lvl)]

        folds = self._folds if self._folds else range(self._n_folds)
        for fold in tqdm(folds, desc=Fore.BLUE + f'Processing {self.archive}' + Style.RESET_ALL):
            train, _, test = data.get_split(fold)
            train_emb, _, test_emb = data.select_splits(fold, embeddings)

            self.fit(train, train_emb)
            predictions = self.predict(test_emb)

            if self._n_neighbors == 1:
                pred_branches = predictions.map(lambda row: row.split(' | '))
            else:
                pred_branches = [row.split(' | ') for row in predictions]

            true_lvls = [test[f'LVL_{i}'].tolist() for i in range(1, self._max_lvl + 1)]
            f1_lvls = []
            for i in range(self._max_lvl):
                if self._n_neighbors == 1:
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


def evaluate(archive, n_neighbors, retrain=False, n_folds=5, folds=None, cuda_device=0, max_lvl=5):
    """ This will perform the following steps:
        1) Get transformer embeddings if:
            a) there are no embeddings under data/amz_metadata/{archive}_emb.pt
            b) flag --retrain is present
        2) Construct the weighted kNN classifier.
        3) Predict the branches on test, and evaluate properly.

        :param archive: str  - name of the archive on which you want to evaluate the method
        :param n_folds: int  - number of folds for cross-validation
        :param folds: list[int]  - (Optional) specified folds, overrides --n_folds.
        :param n_neighbors: int  - number of neighbors for kNN
        :param retrain: bool  - whether to recount the embeddings or not
        :param cuda_device: int  - cuda device for kNN (one per task)
        :param max_lvl: int  - maximum level of hierarchy to consider
    """
    matcher = KNNDeBERTaMatcher(archive, retrain=retrain, n_neighbors=n_neighbors, cuda_device=cuda_device,
                                n_folds=n_folds, folds=folds, max_lvl=max_lvl)
    matcher.evaluate()


if __name__ == '__main__':
    # python src/matchers/m_deberta.py --archive Office_Products --n_neighbors 1 --n_folds 5 --cuda_device 0 --retrain
    fire.Fire(evaluate)
