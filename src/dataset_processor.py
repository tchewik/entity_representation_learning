import gzip
import html
import os
import re

import numpy as np
import pandas as pd
import requests
import tqdm_pathos
from msgspec import Struct
from msgspec.json import Decoder
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

from src.product_dataset import ProductDataset


class Product(Struct):
    title: str
    category: list[str]

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}


class DatasetProcessor:
    def __init__(self, data_path='data/', max_depth=8, rewrite_data=False, n_folds=5, n_cpus=13):
        self.data_path = data_path
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)

        self.dataset_path = os.path.join(self.data_path, 'amz_metadata')
        if not os.path.isdir(self.dataset_path):
            os.mkdir(self.dataset_path)

        self.n_folds = n_folds
        self.folds_path = os.path.join(self.data_path, 'folds')
        if not os.path.isdir(self.folds_path):
            os.mkdir(self.folds_path)

        for i in range(self.n_folds):
            fold_path_i = os.path.join(self.folds_path, f'{i}')
            if not os.path.isdir(fold_path_i):
                os.mkdir(fold_path_i)

        self.max_depth = max_depth
        self._rewrite_data = rewrite_data
        self.n_cpus = n_cpus

        self.url_source = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/'

        self.archive_names = ['Clothing_Shoes_and_Jewelry',
                              #'Books',
                              'Home_and_Kitchen',
                              'Automotive',
                              'Sports_and_Outdoors',
                              'Electronics',
                              #'Toys_and_Games',
                              'Tools_and_Home_Improvement',
                              'Industrial_and_Scientific',
                              #'Cell_Phones_and_Accessories',
                              #'Kindle_Store',
                              #'Office_Products',
                              #'Arts_Crafts_and_Sewing',
                              # 'Patio_Lawn_and_Garden', 'Pet_Supplies', 'Musical_Instruments', 'Video_Games', 'Software'
                              ]  # Sorted by size

    def collect_all(self):
        tqdm_pathos.map(self._collect_all, self.archive_names,
                        dataset_path=self.dataset_path, url_source=self.url_source, _rewrite_data=self._rewrite_data,
                        max_depth=self.max_depth, n_cpus=self.n_cpus)

    @staticmethod
    def _collect_all(archive, dataset_path, url_source, _rewrite_data, max_depth):
        DatasetProcessor._download_archive(archive, dataset_path, url_source)
        dataset = DatasetProcessor._dataset_from_archive(archive, dataset_path, _rewrite_data, max_depth)
        dataset.save_pickle(os.path.join(dataset_path, archive + '.pkl'))

    @staticmethod
    def _download_archive(archive_name, dataset_path, url_source):
        """ Downloads the requested archive from datarepo.eng.ucsd.edu """

        full_archive_name = archive_name + '.json.gz'
        output_path = os.path.join(dataset_path, full_archive_name)
        if os.path.isfile(output_path):
            print(f'File {archive_name}: already downloaded.')
            return

        full_url = url_source + 'meta_' + full_archive_name

        # Streaming, so we can iterate over the response.
        response = requests.get(full_url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True,
                            desc=f'Downloading {archive_name}')
        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise ConnectionError('Could not download the archive: ' + archive_name)

    @staticmethod
    def _dataset_from_archive(archive_name, dataset_path, _rewrite_data, max_depth):
        full_path = os.path.join(dataset_path, archive_name + '.json.gz')
        out_path = full_path[:-8] + '.fth'
        if _rewrite_data == False and os.path.isfile(out_path):
            print(f'File {archive_name}: already preprocessed.')
            return

        json_decoder = Decoder(type=Product)
        data = []
        with gzip.open(full_path) as f:
            for l in tqdm(f, desc=f'Loading {archive_name}'):
                try:
                    data.append(json_decoder.decode(l).to_dict())
                except:
                    pass

        data = DatasetProcessor._clean_up_subset(archive_name, pd.DataFrame(data), max_depth)
        dataset = ProductDataset(data, archive_name)

        # Save all the branches
        concat = data['LVL_1'].astype(str)
        for i in range(1, max_depth):
            concat = concat + ' | ' + data[f'LVL_{i + 1}'].astype(str)

        print(f'Number of unique clean branches in {archive_name}: {concat.unique().shape[0]}.\n')
        dataset.branches = set(concat)

        # Encode classnames for fasttext
        mappers = {f'lvl{i + 1}': DatasetProcessor.make_mapper_demapper(data, i + 1) for i in range(max_depth)}
        dataset.mappers = mappers

        return dataset

    @staticmethod
    def _clean_up_subset(archive_name, data, max_depth):
        all_data_len = data.shape[0]
        print(f'Original length of {archive_name}: {all_data_len}.')

        data = data[data.title != '']
        data = data[data.category.map(len) > 1]  # Keep only entries with annotated hierarchy

        if data.shape[0] == 0:
            print(f'No hierarchical data is found in {archive_name}. Skip this part.\n')
            return data

        ### Preprocess html entities
        html_entities = r'&#?\w{1,16};'
        entities_set = set()
        for text in data.title:
            entities_set.update(re.findall(html_entities, text))

        for text in data.category.map(lambda row: ' '.join(row)):
            entities_set.update(re.findall(html_entities, text))

        entities_to_symb = dict()
        for _ in entities_set:
            entities_to_symb[_] = html.unescape(_)

        def entities_preprocess(text):
            entities_list = re.findall(html_entities, text)

            for entity in entities_list:
                text = re.sub(entity, entities_to_symb[entity], text)
            return text if text else np.nan

        data.title = data.title.apply(entities_preprocess)

        ### Remove duplicates
        data['title_low'] = data.title.map(lambda row: ' '.join(wordpunct_tokenize(row.strip().lower())))
        data = data.drop_duplicates('title_low')
        lengths = data.title_low.map(lambda row: len(row.split()))
        data = data[(lengths > 1) & (lengths < 30)]
        del data['title_low']

        ### Clean up the class hierarchy: keep only repeating subclasses on each level; cut the branch on unknown subclasses
        data.category = data.category.map(lambda row: list(map(entities_preprocess, row)))

        def remove_junk(category):
            if not category or type(category) != str:
                return np.nan

            junk_in_categories = {
                '<span>': '',
                '</span>': '',
                '|': ' '
            }
            for key, value in junk_in_categories.items():
                category = category.replace(key, value)

            return category.lower() if category else np.nan

        data.category = data.category.map(lambda row: list(map(remove_junk, row)))

        def ret_category(arr, n):
            if len(arr) < n + 1:
                return np.nan
            else:
                return arr[n]

        for i in range(max_depth):
            data[f'LVL_{i + 1}'] = data.category.map(lambda row: ret_category(row, i + 1))

        del data['category']

        for i in range(2, max_depth + 1):
            data[f'LVL_{i}'] = data.apply(lambda row: row[f'LVL_{i}'] if type(row[f'LVL_{i - 1}']) == str else np.nan,
                                          axis=1)

        length_threshold = 13
        for i in range(1, max_depth + 1):
            token_counts = data[f'LVL_{i}'].map(lambda row: len(wordpunct_tokenize(row)) if type(row) == str else 0)
            to_remove_long = token_counts[token_counts > length_threshold].index
            for k in range(i, max_depth + 1):
                data.loc[to_remove_long, f'LVL_{k}'] = np.nan

        frequency_threshold = 4
        for j in range(max_depth, -1, -1):
            branch = [i + 1 for i in range(j)]
            for i in branch:
                if i == 1:
                    dcol = data[f'LVL_{i}'].astype(str)
                else:
                    dcol = dcol + ' | ' + data[f'LVL_{i}'].astype(str)

            value_counts = dcol.value_counts()  # Specific column
            to_remove = value_counts[value_counts < frequency_threshold].index  # Remove unfrequents
            for k in range(j + 1, max_depth + 1):
                data.loc[dcol.isin(to_remove), f'LVL_{k}'] = np.nan

        for j in range(max_depth):
            branch = [i + 1 for i in range(j, max_depth)]
            for i in branch:
                if i == 1:
                    dcol = data[f'LVL_{i}'].astype(str)
                else:
                    dcol = dcol + ' | ' + data[f'LVL_{i}'].astype(str)

            value_counts = dcol.value_counts()  # Specific column
            to_remove = value_counts[value_counts < frequency_threshold].index  # Remove unfrequents
            for k in range(j + 1, max_depth + 1):
                data.loc[dcol.isin(to_remove), f'LVL_{k}'] = np.nan

        concat = data['LVL_1'].astype(str)
        for i in range(1, max_depth):
            concat = concat + ' | ' + data[f'LVL_{i + 1}'].astype(str)

        data = data[concat.duplicated(keep=False)]
        data = data[data.LVL_1.isna() == False]
        print(f'Length of deduplicated {archive_name}: {data.shape[0]}.')

        return data.reset_index(drop=True)

    @staticmethod
    def make_mapper_demapper(data, level: int):
        unique_classes = data[f'LVL_{level}'].unique()
        unique_classes = [cl if type(cl) == str else 'nan' for cl in unique_classes]
        _mapper_lvl = dict(zip(unique_classes,
                               [f'LVL{level}_{i}' for i in range(len(unique_classes))]))
        mapper_lvl = {key: value for key, value in _mapper_lvl.items()}
        demapper_lvl = dict(zip(mapper_lvl.values(), mapper_lvl.keys()))

        return mapper_lvl, demapper_lvl

    @staticmethod
    def fasttextize(row, mappers):
        result = str()
        for i in range(1, 6):
            if row[f'LVL_{i}']:
                lvl_label = mappers.get(f'lvl{i}')[0].get(row[f'LVL_{i}'])
                if lvl_label:
                    result += ' __label__' + lvl_label
                else:
                    lvl_label = mappers.get(f'lvl{i}')[0].get('nan')
                    if lvl_label:
                        result += ' __label__' + lvl_label
            else:
                lvl_label = mappers.get(f'lvl{i}')[0].get('nan')
                if lvl_label:
                    result += ' __label__' + lvl_label

        return result

    @staticmethod
    def defasttextize(fasttext_labelstring, mappers):
        result = []
        for i, label in enumerate(fasttext_labelstring.split('__label__')):
            label = label.strip()
            if label:
                result.append(mappers.get(f'lvl{i}')[1].get(label))

        return result


if __name__ == '__main__':
    pr = DatasetProcessor(rewrite_data=True, max_depth=5, n_cpus=15)
    pr.collect_all()
