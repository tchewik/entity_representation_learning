import subprocess

import fire
from tqdm import tqdm

from src.dataset_processor import DatasetProcessor


def run(n_neighbors=5, n_folds=5, cuda_device=0):
    archive_names = DatasetProcessor().archive_names
    for archive in tqdm(archive_names):
        subprocess.run(['python', 'src/matchers/m_fasttext.py',
                        '--archive', archive,
                        '--n_neighbors', str(n_neighbors),
                        '--n_folds', str(n_folds),
                        '--cuda_device', str(cuda_device)])


if __name__ == '__main__':
    fire.Fire(run)
