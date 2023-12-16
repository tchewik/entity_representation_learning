import os

import fire
import numpy as np
import torch
from allennlp.predictors import Predictor
from colorama import Fore, Style
from nltk.tokenize import wordpunct_tokenize
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.extmath import weighted_mode
from torch.nn.functional import normalize
from tqdm import tqdm

from src.hierarchical_classifier import HierarchicalClassifier
from src.product_dataset import ProductDataset

tqdm.pandas()


class KNNDeBERTaCMatcher(HierarchicalClassifier):
    def __init__(self, archive: str,
                 retrain=False,
                 batch_size=20,
                 model_dir='models/c_deberta',
                 dataset_path='data/amz_metadata',
                 folds_path='data/folds',
                 n_folds=5,
                 folds=None,
                 n_neighbors=1,
                 max_lvl=5,
                 cuda_device=-1):
        """
            :param archive: str  - Name of the corpus part.
            :param retrain: bool  - When set to False (default), only evaluate the previously collected embeddings.
            :param batch_size: int  - Batch size for embeddings collection.
            :param model_dir: str  - Directory with the fine-tuned models.
            :param dataset_path: str  - Path to the corpus directory.
            :param folds_path: str  - Path to the directory with information about folds.
            :param n_folds: int (optional)  - Number of folds.
            :param folds: list[int] - (optional) Specific folds. This parameter overrides --n_folds if present.
            :param n_neighbors: int  - Number of neighbors to consider in kNN.
            :param max_lvl: int (optional)  - Depth of the hierarchy to consider.
            :param cuda_device: int  - Number of the GPU to use; -1 for CPU-only.
        """
        super().__init__(archive=archive, dataset_path=dataset_path,
                         folds_path=folds_path, n_folds=n_folds, folds=folds, max_lvl=max_lvl)

        self._model_dir = os.path.join(model_dir, self.archive)
        assert os.path.isdir(self._model_dir), 'Model is not found. Train the classifier beforehand!'

        self._n_neighbors = n_neighbors
        self._retrain = retrain
        self._batch_size = batch_size

        self._cuda_int = cuda_device
        cuda = 'cpu' if cuda_device == -1 else f'cuda:{cuda_device}'
        self._cuda_device = torch.device(cuda)

    def _get_tokens(self, data):
        return data.title.map(lambda row: ' '.join(wordpunct_tokenize(row.lower())))

    def fit(self, data, train_embeddings):
        y = data['LVL_1'].astype(str)
        for i in range(2, self._max_lvl + 1):
            y += ' | ' + data[f'LVL_{i}'].astype(str)

        self._known_embeddings = normalize(train_embeddings).half().to(self._cuda_device)
        self._known_labels = y

    def predict(self, text_emb: torch.Tensor):
        """
        :param text_emb: Embeddings of the input texts
        :return: Label derived with kNN
        """

        results = []  # found labels
        text_emb = normalize(text_emb).half().to(self._cuda_device)
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

    def collect_embeddings(self, data, fold):
        predictor = Predictor.from_path(archive_path=os.path.join(self._model_dir, f'fold_{fold}', 'model.tar.gz'),
                                        cuda_device=self._cuda_int)
        predictor._model = predictor._model.to(self._cuda_device)
        predictor._model._device = self._cuda_device

        def get_emb_batch(batch):
            predictions = predictor.predict_batch_json([{'sentence': text.lower()} for text in batch])
            return [prediction['text_repr'] for prediction in predictions]

        def process_batches(texts, batch_size):
            def chunker(seq, size):
                for pos in range(0, len(seq), size):
                    yield seq.iloc[pos:min(pos + size, seq.shape[0])]

            result = []
            description = f'Getting finetuned embeddings for {self.archive}' + f' (GPU {self._cuda_int})' * (
                        self._cuda_int > -1)
            for batch in tqdm(chunker(texts, batch_size),
                              desc=Fore.BLUE + description + Style.RESET_ALL,
                              total=texts.shape[0] // batch_size + 1):
                result += get_emb_batch(batch)

            return result

        # Concatenate batch results
        all_embeddings = process_batches(data.title, batch_size=self._batch_size)
        embeddings = torch.Tensor(all_embeddings).to(self._cuda_device)
        torch.save(embeddings, os.path.join(self._dataset_path, self.archive + f'_emb_{fold}_tuned.pt'))

        return embeddings

    def evaluate(self):

        data = ProductDataset.from_pickle(os.path.join(self._dataset_path, self.archive + '.pkl'))

        all_f1s_lvls = [[] for i in range(self._max_lvl)]
        f1s_avg = []
        accuracy_lvls = [[] for _ in range(self._max_lvl)]

        folds = self._folds if self._folds else range(self._n_folds)
        for fold in tqdm(folds, desc=Fore.BLUE + f'Processing {self.archive}' + Style.RESET_ALL):

            embeddings_path = os.path.join(self._dataset_path, self.archive + f'_emb_{fold}_tuned.pt')
            if not os.path.isfile(embeddings_path) or self._retrain:
                embeddings = self.collect_embeddings(data.df, fold)
            else:
                embeddings = torch.load(embeddings_path, map_location=self._cuda_device)

            train, _, test = data.get_split(fold)
            train_emb, _, test_emb = data.select_splits(fold, embeddings)

            self.fit(train, train_emb)
            predictions = self.predict(test_emb)
            pred_branches = [row.split(' | ') for row in predictions]

            true_lvls = [test[f'LVL_{i}'].tolist() for i in range(1, self._max_lvl + 1)]
            f1_lvls = []
            for i in range(self._max_lvl):
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


def evaluate(archive, n_neighbors=1, retrain=False, n_folds=5, folds=None, cuda_device=-1, max_lvl=5, batch_size=32):
    """ This will perform the following steps:
        1) Get FINETUNED transformer embeddings if:
            a) there are no embeddings under data/amz_metadata/{archive}_emb_N_tuned.pt
            b) flag --retrain is present
        2) Construct the weighted kNN classifier.
        3) Predict the branches on test, and evaluate properly.

        :param archive: str  - name of the archive on which you want to evaluate the method
        :param n_folds: int  - number of folds for cross-validation
        :param folds: list[int]  - (Optional) specified folds, overrides --n_folds.
        :param n_neighbors: int  - number of neighbors for kNN
        :param retrain: bool  - whether to recount the embeddings or not
        :param cuda_device: int  - cuda device for embeddings collection and kNN (one per task)
        :param max_lvl: int  - maximum level of hierarchy to consider
        :param batch_size: int  - Batch size for embeddings collection.
    """
    matcher = KNNDeBERTaCMatcher(archive, retrain=retrain, n_neighbors=n_neighbors, cuda_device=cuda_device,
                                 n_folds=n_folds, folds=folds, max_lvl=max_lvl, batch_size=batch_size)
    matcher.evaluate()


if __name__ == '__main__':
    # python src/matchers/m_deberta1-5.py Industrial_and_Scientific --n_neighbors 1 --n_folds 5 --cuda_device 0 --retrain
    fire.Fire(evaluate)
