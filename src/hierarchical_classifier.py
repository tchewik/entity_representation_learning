import itertools


class HierarchicalClassifier:
    def __init__(self, archive, dataset_path='data/amz_metadata', folds_path='data/folds',
                 n_folds=5, max_lvl=8, folds=None):
        self.archive = archive
        self._dataset_path = dataset_path
        self._folds_path = folds_path
        self._n_folds = n_folds
        self._folds = folds
        self._max_lvl = max_lvl

    def decode_class(self, fasttext_mapper, classname: str, lvl: int):
        out_of_lvl = fasttext_mapper.get(f'lvl{lvl}')[0].get('nan', 'nan')
        return fasttext_mapper.get(f'lvl{lvl}')[0].get(classname, out_of_lvl)

    def all_instances_of_class_from_branch(self, fasttext_mapper, branch: list, lvl: int):
        out_of_lvl = fasttext_mapper.get(f'lvl{lvl}')[0].get('nan', 'nan')
        if f'LVL{lvl}' not in ' '.join(branch):
            return [out_of_lvl]

        res = []
        for label in branch:
            if f'LVL{lvl}' in label:
                res.append(label.replace('__label__', ''))

        return res

    def class_from_branch(self, fasttext_mapper, branch: str, lvl: int, sep=None):
        out_of_lvl = fasttext_mapper.get(f'lvl{lvl}')[0].get('nan', 'nan')
        if f'LVL{lvl}' not in branch:
            return out_of_lvl

        items = branch.split(sep) if sep else branch.split()
        for label in items:
            if f'LVL{lvl}' in label:
                return label.replace('__label__', '')

        return out_of_lvl

    def decode_branches(self, branches, sep=None):
        lvls = []
        for i in range(self._max_lvl):
            lvls = [self.class_from_branch(line, lvl=i + 1, sep=sep) for line in branches]

        return lvls

    @staticmethod
    def _match_with_standard(sequence, standard_branches):
        return ' | '.join(sequence) in standard_branches

    def best_sequence(self, predictions, standard_branches, fasttext_mapper):
        """ Finds the realistic branch (levels path) across the predictions.
            If value for level is not found among the predictions, it's default value is 'nan'.

            (!) If no partially realistic branch is found, revert to the best_sequence_simple.
        """
        nans = [fasttext_mapper.get(f'lvl{i + 1}')[0].get('nan', 'nan') for i in range(self._max_lvl)]

        for i in range(1, self._max_lvl):
            if not predictions[i]:
                predictions[i] = [nans[i]]

        for i in range(self._max_lvl, -1, -1):
            for j in range(i + 1):
                pool = [predictions[k][:-1] for k in range(j)]

            for j in range(i, self._max_lvl):
                pool += [predictions[k] for k in range(j, self._max_lvl)]

            for branch in itertools.product(*pool):
                if self._match_with_standard(branch, standard_branches):
                    return branch

        return self._best_sequence_simple(predictions, standard_branches, fasttext_mapper)

    def _best_sequence_simple(self, predictions, standard_branches, fasttext_mapper):
        """ Finds the realistic branch (levels path) across the predictions.
            First pass is for seq,
            Second pass is for seq[:-1] + ['nan'],
            The third, for seq[:-2] + ['nan', 'nan'], etc.
        """

        nans = [fasttext_mapper.get(f'lvl{i + 1}')[0].get('nan', 'nan') for i in range(self._max_lvl)]

        previous_cutted_seq = None
        for i in range(self._max_lvl, 0, -1):
            cutted_seq = predictions[:i] + [[nan] for nan in nans[i:self._max_lvl]]
            if cutted_seq != previous_cutted_seq:
                previous_cutted_seq = cutted_seq
                for seq in itertools.product(*cutted_seq):
                    if self._match_with_standard(seq, standard_branches):
                        return seq

        for seq in itertools.product(*predictions):
            return list(seq)
