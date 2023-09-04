import os
import numpy as np

def load(file, encoding='utf-8'):
    """load a file in format: <subject> ,<category>"""
    with open(file, 'r', encoding=encoding) as fd:
        datas = fd.read()
    datas = datas.splitlines()

    results = {}
    for data in datas:
        subject, category = data.split(' ,')
        x = results.get(subject, set())
        x.add(category)
        results[subject] = x
    return results

def write(file, dict, encoding='utf-8'):
    datas = []
    for subject in sorted(dict.keys()):
        for category in sorted(dict[subject]):
            datas.append('%s ,%s\n' % (subject, category))

    with open(file, 'w', encoding=encoding) as fd:
        fd.writelines(datas)

def is_LIWC_data(data):
    """data is {string: set/list}"""
    if not isinstance(data, dict):
        return False

    if not np.all([isinstance(i, str) for i in data.keys()]) or \
       not np.all([np.iterable(i) for i in data.values()]):
        return False

    return True

def categories_to_score(categories, score_table=None):
    # posemo (Positive Emotions) and negemo (Negative Emotions) can exist as the same time
    if score_table is None:
        score_table = {
            'posemo (Positive Emotions)': 1,
            'negemo (Negative Emotions)': -1
        }
    else:
        if not isinstance(score_table, dict):
            raise ValueError('score_table should be a dictionary')

    categories = set(categories)
    scores = [score_table.get(category, 0) for category in categories]
    return sum(scores)

class LingusticsInquiry(object):
    # refer to https://github.com/EricWiener/liwc-analysis/blob/master/liwcanalysis/liwcanalysis.py

    def __init__(self, file_or_dict={}):
        if isinstance(file_or_dict, str):
            data = load(file_or_dict)
        else:
            data = file_or_dict

        if not is_LIWC_data(data):
            raise ValueError()

        self._words = {i: j for i, j in data.items() if '*' not in i}
        self._roots = {i[:-1]: j for i, j in data.items() if '*' in i}

    @property
    def words(self):
        return list(self._words.keys())

    @property
    def roots(self):
        return list(self._roots.keys())

    @property
    def categories(self):
        X = set()
        for i in self._words.values():
            X.update(i)
        for i in self._roots.values():
            X.update(i)
        return X

    def merge_data(self, data):
        if not is_LIWC_data(data):
            raise ValueError(
                'Incorrect format, please check is_LIWC_data function.')

        for subject, categories in data.items():
            D = self._roots if '*' in subject else self._words
            subject = subject.replace('*', '')

            x = D.get(subject, set())
            x.update(categories)
            D[subject] = x

    def add_dictionaries(self, *files_or_dicts):
        for file_or_dict in files_or_dicts:
            data = load(file_or_dict) if isinstance(file_or_dict, str) \
                                      else file_or_dict
            self.merge_data(data)

    def get_categories(self, word):
        if not isinstance(word, str):
            raise ValueError()

        if word in self._words:
            return self._words.get(word)

        N = len(word)
        x = 0
        while x < N:
            _word = word[:None] if x == 0 else word[:-x]
            if _word not in self._roots:
                x += 1
                continue
            return self._roots.get(_word)

        return set()
