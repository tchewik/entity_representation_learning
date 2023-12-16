import subprocess

import fire
from tqdm import tqdm

from src.dataset_processor import DatasetProcessor


def run(retrain=False):
    archive_names = DatasetProcessor().archive_names
    for archive in tqdm(archive_names):
        with open(f'c_fasttext_{archive}.out', 'w') as f:
            if archive in ['Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Industrial_and_Scientific']:
                lr = 0.2
            else:
                lr = 1.0

            # Call script and capture stdout
            p = subprocess.Popen(['python', 'src/classifiers/c_fasttext.py',
                                  '--archive', archive,
                                  '--retrain', str(retrain),
                                  '--lr', str(lr)],
                                 stdout=subprocess.PIPE)

            # Write output to file
            for line in p.stdout:
                f.write(line.decode('utf-8'))

            p.wait()


if __name__ == '__main__':
    fire.Fire(run)
