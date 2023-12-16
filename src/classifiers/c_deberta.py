import gzip
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from functools import partial
from multiprocessing import Pool, Queue

import fire
import msgspec
import numpy as np
import pandas as pd
from colorama import Fore, Style
from msgspec import Struct
from msgspec.json import Decoder
from parallelbar import progress_map
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import mykeys
from src.hierarchical_classifier import HierarchicalClassifier
from src.product_dataset import ProductDataset

os.environ["WANDB_API_KEY"] = mykeys.wandb_api_key
TRAIN_CONFIG = 'configs/deberta1-5.jsonnet'

_gpu_queue = Queue()
_all_processes = []


class Prediction(Struct):
    probs1: list[float]
    probs2: list[float]
    probs3: list[float]
    probs4: list[float]
    probs5: list[float]
    label1: str
    label2: str
    label3: str
    label4: str
    label5: str

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}


_json_decoder = Decoder(type=Prediction)


def _train(fold, num_labels_i, archive, pretrained_model, model_dir, lr, batch_size):
    gpu = _gpu_queue.get()  # Get next free GPU

    try:
        print(f"Training {archive} fold {fold} on GPU {gpu}")

        output_dir = os.path.join(model_dir, f'fold_{fold}')
        overrides = {
            "dataset_name": archive,
            "pretrained_model": pretrained_model,
            "cuda_device": gpu,
            "lr": lr,
            "batch_size": batch_size,
            "foldnum": fold
        }
        overrides.update({f"num_labels{i + 1}": num_labels_i[i] for i in range(5)})

        for key, value in overrides.items():
            os.environ[key] = str(value)

        log_file = f"log_train_gpu{gpu}.txt"
        err_file = f"err_train_gpu{gpu}.txt"

        with open(log_file, 'w') as out, open(err_file, 'w') as err:
            p = subprocess.Popen(
                ["allennlp", "train", "-s", output_dir, TRAIN_CONFIG],
                env=os.environ,
                stdout=out,
                stderr=err
            )

    except Exception as e:
        print(f"Training failed on GPU {gpu}: {e}")

    finally:
        _gpu_queue.put(gpu)


def _predict(fold, archive, model_dir):
    gpu = _gpu_queue.get()
    print(f"Predicting {archive} fold {fold} on GPU {gpu}")

    try:
        # log_file = f"log_predict_gpu{gpu}_train.txt"
        # err_file = f"err_predict_gpu{gpu}_train.txt"
        #
        # with open(log_file, 'w') as out, open(err_file, 'w') as err:
        #     p = subprocess.Popen(
        #         ['allennlp', 'predict', '--use-dataset-reader', '--silent',
        #          '--cuda-device', str(gpu),
        #          '--output-file', f'{model_dir}/fold_{fold}/{archive}_predictions_train.json',
        #          f'{model_dir}/fold_{fold}/model.tar.gz', f'data/folds/{fold}/{archive}_train_c_deberta_tokens.json'],
        #         stdout=out,
        #         stderr=err
        #     )

        log_file = f"log_predict_gpu{gpu}_test.txt"
        err_file = f"err_predict_gpu{gpu}_test.txt"

        with open(log_file, 'w') as out, open(err_file, 'w') as err:
            cuda_device = ['--cuda-device', str(gpu)] if gpu > -1 else []
            p = subprocess.Popen(
                ['allennlp', 'predict', '--use-dataset-reader', '--silent'] + cuda_device + [
                    '--output-file', f'{model_dir}/fold_{fold}/{archive}_predictions_test.json',
                    f'{model_dir}/fold_{fold}/model.tar.gz', f'data/folds/{fold}/{archive}_test_c_deberta_tokens.json'],
                stdout=out,
                stderr=err
            )

    except Exception as e:
        print(f"Prediction failed on GPU {gpu}: {e}")

    finally:
        _gpu_queue.put(gpu)


class DeBERTaClassifier(HierarchicalClassifier):
    def __init__(self, archive: str,
                 n_top_nodes=8,
                 retrain=False,
                 run_predict=False,
                 model_dir='models/c_deberta',
                 pretrained_model='microsoft/deberta-v3-base',
                 dataset_path='data/amz_metadata',
                 folds_path='data/folds',
                 n_folds=5,
                 max_lvl=5,
                 lr=2e-5,
                 batch_size=5,
                 available_gpus=None,
                 one_task_per_gpu=True,
                 continue_training=False,
                 threads=10,
                 ):
        """
            :param archive: str  - Name of the corpus part.
            :param n_top_nodes: int  - Number of predicted class labels to consider in the branch construction.
            :param retrain: bool  - When set to False (default), only evaluate the previously trained models.
            :param pretrained_model_path: str  - Full path to the `*.bin` pretrained fastText model.
            :param dataset_path: str  - Path to the corpus directory.
            :param folds_path: str  - Path to the directory with information about folds.
            :param n_folds: int  - (optional) Number of folds.
            :param max_lvl: int  - (optional) Maximum level of hierarchy.
            :param threads: int  - (optional) Number of CPU threads for parallelized stuff.
        """

        super().__init__(archive=archive, dataset_path=dataset_path,
                         folds_path=folds_path, n_folds=n_folds, max_lvl=max_lvl)

        self._n_top_nodes = n_top_nodes
        self._retrain = retrain
        self._continue_training = continue_training

        self.num_labels_i = []

        self._model_dir = os.path.join(model_dir, self.archive)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        if self._retrain and not self._continue_training:
            shutil.rmtree(self._model_dir, ignore_errors=True)
            os.mkdir(self._model_dir)

        self._run_predict = run_predict

        if not os.path.isdir(self._model_dir):
            os.mkdir(self._model_dir)

        self._pretrained_model = pretrained_model
        self._lr = lr
        self._batch_size = batch_size

        if not available_gpus:
            available_gpus = [0, 1]
        self._available_gpus = available_gpus
        self._one_task_per_gpu = one_task_per_gpu

        self._completed_folds = 0

        self._gpu_queue = Queue()

        self._threads = threads

    def evaluate(self):
        data = ProductDataset.from_pickle(os.path.join(self._dataset_path, self.archive + '.pkl'))

        # Run allennlp train using available resources if --retrain  #####
        if self._retrain:
            self.prepare_data(data)
            self.num_labels_i = data.unique_classes()
            self.fit()

        # Run allennlp predict using available resources if *_predictions_test.json files are missing  ######
        if self._run_predict:
            self.run_prediction()

        # Do the predictions zipping in parallel over all folds if *.json.gz files are missing  ######
        folds = self._folds if self._folds else range(self._n_folds)
        subdirs = [os.path.join(self._model_dir, f'fold_{fold}') for fold in folds]
        predictions_paths = [os.path.join(subdir, f'{self.archive}_predictions_test.json') for subdir in subdirs]
        output_paths = [os.path.join(subdir, 'predictions_test.json.gz') for subdir in subdirs]
        missing_zipped_predictions = [(fold, inpath, outpath) for fold, (inpath, outpath) in enumerate(
            zip(predictions_paths, output_paths)) if not os.path.isfile(outpath)]
        progress_map(self.gzip_the_predictions, missing_zipped_predictions,
                     n_cpu=len(missing_zipped_predictions))

        # Evaluate the classifier  #####
        f1s_lvls = [[] for _ in range(self._max_lvl)]
        f1s_avg = []
        accuracy_lvls = [[] for _ in range(self._max_lvl)]

        for fold in tqdm(folds, desc=Fore.BLUE + f'Processing {self.archive}' + Style.RESET_ALL):
            _, _, test = data.get_split(fold)
            true_lvls = [test[f'LVL_{i}'].tolist() for i in range(1, self._max_lvl + 1)]

            subdir = os.path.join(self._model_dir, f'fold_{fold}')

            all_classnames = [
                [cn.strip()
                 for cn in open(os.path.join(subdir, 'vocabulary', f'labels{i + 1}.txt'), 'r').readlines() if cn]
                for i in range(self._max_lvl)]

            test_predictions_path = os.path.join(subdir, 'predictions_test.json.gz')

            all_lines = []
            with gzip.open(test_predictions_path, 'rb') as file:
                for line in tqdm(file.readlines(), desc='Reading zipped predictions', leave=False):
                    all_lines.append(_json_decoder.decode(line).to_dict())

            with multiprocessing.Pool(processes=self._threads) as pool:
                load_nn_preds_func = partial(self.load_nn_preds,
                                             all_classnames=all_classnames,
                                             _n_top_nodes=self._n_top_nodes)
                enum_alllines = list(enumerate(all_lines))
                results = []
                with tqdm(total=len(enum_alllines), desc='Resolving nn predictions from dicts', leave=False) as pbar:
                    for result in pool.imap(load_nn_preds_func, enum_alllines):
                        res = [result[0], [[label for (label, prob) in level] for level in result[1]]]
                        results.append(res)
                        pbar.update()

                best_seq_inputs = [r for i, r in sorted(results, key=lambda x: x[0])]  # ordered results

            with multiprocessing.Pool(processes=self._threads) as pool:
                bestseq_func = partial(self.best_sequence,
                                       standard_branches=data.branches,
                                       mappers=data.mappers,
                                       max_lvl=self._max_lvl)
                enum_bsi = list(enumerate(best_seq_inputs))
                results = []
                with tqdm(total=len(enum_bsi), desc=f'Decoding level paths, n={self._n_top_nodes}',
                          leave=False) as pbar:
                    for result in pool.imap(bestseq_func, enum_bsi):
                        results.append(result)
                        pbar.update()

                pred_branches = [r for i, r in sorted(results, key=lambda x: x[0])]  # ordered results

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

    @staticmethod
    def best_sequence(pred_lvls, standard_branches, mappers, max_lvl):
        """ This monstrosity is for ordered multiprocessing of _best_sequence_simple """
        h = HierarchicalClassifier(archive=None, max_lvl=max_lvl)
        return pred_lvls[0], h._best_sequence_simple(pred_lvls[1], standard_branches, mappers)

    def prepare_data(self, data):
        """ Sets up *.json files for the custom allennlp dataset reader (allennlp train/predict) """

        def formatted_df(data, output_file):
            df = pd.DataFrame({
                'text': data.title.map(lambda row: row.lower()),
                'label1': data['LVL_1'].astype(str),
                'label2': data['LVL_2'].astype(str),
                'label3': data['LVL_3'].astype(str),
                'label4': data['LVL_4'].astype(str),
                'label5': data['LVL_5'].astype(str),
            })

            with open(output_file, 'w') as fp:
                fp.write('\n'.join(json.dumps(i) for i in df.to_dict('records')) + '\n')

        for fold in tqdm(range(self._n_folds),
                         desc=Fore.BLUE + f'Preparing {self.archive} for training' + Style.RESET_ALL):
            train, val, test = data.get_split(fold, fill_unknown_classes=True)

            formatted_df(train, os.path.join(self._folds_path, str(fold),
                                             f'{self.archive}_train_c_deberta_tokens.json'))
            formatted_df(val, os.path.join(self._folds_path, str(fold), f'{self.archive}_val_c_deberta_tokens.json'))
            formatted_df(test, os.path.join(self._folds_path, str(fold), f'{self.archive}_test_c_deberta_tokens.json'))

    @staticmethod
    def gzip_the_predictions(fold_n_paths):
        """ Reduces size of enormous *.json prediction files produced by allennlp.
            Produces small *.json.gz file with predictions (rounded probas, labels, etc.)
            After *.json.gz is produced, original file can be safely removed.

            :param fold_n_paths: tuple(int, str, str)  - fold number and paths to the input *.json file and output *.json.gz
        """
        fold, inpath, outpath = fold_n_paths
        with open(inpath) as f_in:
            with gzip.open(outpath, 'wb') as f_out:
                for line in tqdm(f_in.readlines(), desc=f'Reducing size of the json predictions (fold {fold}).'):
                    line = _json_decoder.decode(line).to_dict()

                    if not line:
                        break

                    keys = [key for key in line.keys() if key.startswith('probs')]
                    for key in keys:
                        line[key] = [round(num, 6) for num in line[key]]

                    # json.dump(line, f_out)
                    f_out.write(msgspec.json.encode(Prediction(**line)))
                    f_out.write(b"\n")

    @staticmethod
    def split_by_size(some_list, n):
        """ Splits list by chunks of length n """
        for i in range(0, len(some_list), n):
            yield some_list[i:i + n]

    @staticmethod
    def spinner_check_for_file_creation(filepath, speed=0.5):
        """ Tracks creation of a given file. Loop stops when the file is created (obv. by another process).

            :param filepath: File to track
            :param speed: Speed of the spinning thing, equals to the timeout of tracking
        """
        spinner = ['|', '/', '—', '\\']
        sp_index = 0
        while not os.path.isfile(filepath):
            sys.stdout.write("\r" + spinner[sp_index])
            sys.stdout.flush()
            sp_index = (sp_index + 1) % len(spinner)
            time.sleep(speed)

    @staticmethod
    def spinner_check_for_file_editing(filepath, checkspeed=60, speed=0.1):
        """ Tracks continuous changes in a given file. Loop stops when the file stops changing.

            :param filepath: str  - File to track changes
            :param checkspeed: int  - Timeout (in seconds) of tracking
            :param speed: float  - Speed of the spinning thing
        """
        DeBERTaClassifier.spinner_check_for_file_creation(filepath, speed)

        spinner = ['|', '/', '—', '\\']
        sp_index = 0
        prev_size = None
        prev_timestamp = None
        last_time_check = time.time()
        while True:
            # This part with file size checking goes only once in a minute
            time_check = time.time()
            if time_check - last_time_check >= checkspeed:
                last_time_check = time_check
                curr_size = os.path.getsize(filepath)
                curr_timestamp = os.path.getmtime(filepath)

                if prev_size is not None and prev_timestamp is not None:
                    if curr_size != prev_size or curr_timestamp > prev_timestamp:
                        prev_size = curr_size
                        prev_timestamp = curr_timestamp
                    else:
                        break

                else:
                    # Initial check
                    prev_size = curr_size
                    prev_timestamp = curr_timestamp

            sys.stdout.write("\r" + spinner[sp_index])
            sys.stdout.flush()
            sp_index = (sp_index + 1) % len(spinner)
            time.sleep(speed)

    def fit(self):
        """ This method fits the model on each cross validation fold.
            Details:
            *  Spreads the models on each CV fold over the available gpus. Depends on --gpus.
            *  If --one_task_per_gpu is set to False, models for each fold train simultaneously using all given gpus.
            *  As for the spinning thing, there is a simple solution to wait while the use of gpu ends
               by tracking when the model.tar.gz file is appearing.
        """

        for gpu in self._available_gpus:
            _gpu_queue.put(gpu)

        kwargs = {
            'archive': self.archive,
            'pretrained_model': self._pretrained_model,
            'model_dir': self._model_dir,
            'lr': self._lr,
            'batch_size': self._batch_size,
        }
        train_partial = partial(_train, **kwargs)
        folds = [i for i in range(self._n_folds)]
        folds_and_labels = [(fold, self.num_labels_i) for fold in folds]
        if self._continue_training:
            folds_and_labels = [fai for fai in folds_and_labels
                                if not os.path.isfile(os.path.join(self._model_dir, f'fold_{fai[0]}', 'model.tar.gz'))]

        batches = self.split_by_size(folds_and_labels, len(self._available_gpus)) if self._one_task_per_gpu else [
            folds_and_labels]

        for batch in batches:
            with Pool(len(self._available_gpus), maxtasksperchild=1) as pool:

                pool.starmap(train_partial, batch)

                processes = multiprocessing.active_children()
                while len(processes) > 0:
                    for p in processes:
                        if not p.is_alive():
                            processes.remove(p)

                    time.sleep(10)

                # Here, wait for all batch to finish
                for (fold, label) in batch:
                    self.spinner_check_for_file_creation(os.path.join(self._model_dir, f'fold_{fold}', 'model.tar.gz'))

        print("Trained on all the folds!")

    def run_prediction(self):
        """ This method produces test predictions of the model on each cross validation fold.
            Details:
            *  Spreads the models on each CV fold over the available gpus. Depends on --gpus.
            *  If --one_task_per_gpu is set to False, models for each fold predict on test simultaneously
               using all given gpus.
            *  As for the spinning thing, there is a simple solution to wait while the use of gpu ends
               by tracking when the *.json prediction file stops updating.
        """

        for gpu in self._available_gpus:
            _gpu_queue.put(gpu)

        # Predict the classes and embeddings into *.json files
        kwargs = {
            'archive': self.archive,
            'model_dir': self._model_dir,
        }
        predict_partial = partial(_predict, **kwargs)
        folds = [i for i in range(self._n_folds)]
        batches = self.split_by_size(folds, len(self._available_gpus)) if self._one_task_per_gpu else [folds]

        for batch in batches:

            with Pool(len(self._available_gpus)) as pool:

                pool.map(predict_partial, batch)

                # Here, wait for all batch to finish
                for fold in batch:
                    self.spinner_check_for_file_editing(
                        os.path.join(self._model_dir, f'fold_{fold}', f'{self.archive}_predictions_test.json'))

            print("Predicted on all the folds!")

    @staticmethod
    def load_nn_preds(numbered_line, all_classnames, _n_top_nodes):
        """ This method:
            1. Collects {label: proba} dictionary from the line of *.json allennlp prediction and model vocabulary.
            2. Filters out the predictions in the dictionary keeping only _n_top_nodes (including top-1 for each level).
        """
        number, l = numbered_line

        all_zips = [[(l[f'label{i + 1}'], max(l[f'probs{i + 1}']))] for i in range(5)]
        all_predictions = [dict(zip(all_classnames[i], l[f'probs{i + 1}'])) for i in range(5)]

        all_probs = [prob for level in all_predictions for (label, prob) in level.items()]
        min_acceptable = sorted(all_probs, key=lambda x: -x)[_n_top_nodes]

        all_zips[0] += [(key, value) for key, value in sorted(
            all_predictions[0].items(), key=lambda item: -item[1]) if
                        not (key, value) in all_zips[0] and value > min_acceptable]

        all_zips[1] += [(key, value) for key, value in sorted(
            all_predictions[1].items(), key=lambda item: -item[1]) if
                        not (key, value) in all_zips[1] and value > min_acceptable]

        all_zips[2] += [(key, value) for key, value in sorted(
            all_predictions[2].items(), key=lambda item: -item[1]) if
                        not (key, value) in all_zips[2] and value > min_acceptable]

        all_zips[3] += [(key, value) for key, value in sorted(
            all_predictions[3].items(), key=lambda item: -item[1]) if
                        not (key, value) in all_zips[3] and value > min_acceptable]

        all_zips[4] += [(key, value) for key, value in sorted(
            all_predictions[4].items(), key=lambda item: -item[1]) if
                        not (key, value) in all_zips[4] and value > min_acceptable]

        return number, all_zips


def evaluate(archive, n_folds=5, retrain=False, continue_training=False, run_predict=False,
             n_top_nodes=8, max_lvl=5,
             lr=2e-5, batch_size=5, gpus='0,1', one_task_per_gpu=True, threads=10):
    """ This will perform the following steps:
        1) Construct the classifier
        2) Train if:
            a) there are no models under models/c_deberta/{archive}/fold_*/
            b) flag --retrain is present
            c) there are only some of the folds present under models/{archive}/fold_*/ and flag --continue_training is True
        3) Run allennlp prediction to obtain *.json predictions for each fold if:
            a) there are no predictions under model/{archive}/fold_*/
            b) flag --run_predict is present
        4) Grab the *.json predictions, construct final predictions (branches), and evaluate properly.

        :param archive: str  - name of the archive on which you want to evaluate the method
        :param n_folds: int  - number of folds for cross-validation
        :param retrain: bool  - whether to retrain the model or not
                                (WARNING: retrain=True with continue_training=False removes the trained model's dir)
        :param continue_training: bool  - whether to continue training preserving existing folds or not
        :param run_predict: bool  - whether to (re)run prediction or not; requires existing models or retrain=True
        :param n_top_nodes: int  - number of top predicted nodes to consider for branch construction
        :param max_lvl: int  - maximum level of hierarchy to consider
        :param lr: float  - learning rate for training the model
        :param gpus: str or tuple or list or int  - available GPUs for the tasks (typically int: 0 or list: [4,5,6])
        :param one_task_per_gpu: bool  - whether to use one task per GPU or slam all the tasks onto available GPUs at once
        :param batch_size: int  - (optional) Size of a batch for training/prediction.
        :param threads: int  - (optional) Number of CPU threads for parallelized stuff.
    """

    if type(gpus) == tuple:
        available_gpus = list(gpus)
    elif type(gpus) == str:
        available_gpus = list(map(int, gpus.split(',')))
    elif type(gpus) == int:
        available_gpus = [gpus]
    else:
        available_gpus = gpus

    clf = DeBERTaClassifier(archive,
                            n_top_nodes=n_top_nodes, retrain=retrain, run_predict=run_predict,
                            n_folds=n_folds, max_lvl=max_lvl, lr=lr, batch_size=batch_size,
                            available_gpus=available_gpus, one_task_per_gpu=one_task_per_gpu,
                            continue_training=continue_training, threads=threads)
    clf.evaluate()


if __name__ == '__main__':
    # python src/classifiers/c_deberta.py Electronics -b 128 --n_top_nodes 7
    fire.Fire(evaluate)
