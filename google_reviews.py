import os
import copy
import json
import tqdm
import pickle
import hashlib
import numpy as np
import pandas as pd

import utils
import liwc_data

_review_data = {}
_path_to_digest = {}
_name_to_pathes = {}

def set_review_data(dict_or_json):
    if isinstance(dict_or_json, str):
        with open(dict_or_json, 'r', encoding='utf-8') as fd:
            data = json.load(fd)
    else:
        data = dict_or_json

    if not isinstance(data, dict):
        raise ValueError

    global _review_data, _path_to_digest, _name_to_pathes

    _review_data = data
    _path_to_digest = {j['path']: i for i, j in _review_data.items()}
    for i in data.values():
        name = i['name']
        path = i['path']
        x = _name_to_pathes.get(name, [])
        x.append(path)
        _name_to_pathes[name] = x

set_review_data(
    os.path.join(os.path.split(__file__)[0], 'review_data.json'))

def save_review_data(file, review_data):
    with open(file, 'w', encoding='utf-8') as fd:
        json.dump(review_data, fd, ensure_ascii=False, indent=4)

def get_review_info(path):
    assert(os.path.isfile(path))
    name = os.path.split(path)[-1][:-4]
    with open(path, 'rb') as fd:
        digest = hashlib.sha256(fd.read()).hexdigest()
    return name, digest

def discover_review_data(root):
    pickle_files = utils.discover_pickles(root)

    review_data = {}
    for pickle_file in tqdm.tqdm(pickle_files, position=0):
        name, digest = get_review_info(pickle_file)
        assert(digest not in review_data)
        review_data[digest] = {'name': name, 'path': pickle_file}
    return review_data

def review_data_identifier(hash_or_name_or_path):
    key = hash_or_name_or_path
    assert(isinstance(key, str))

    global _review_data, _path_to_digest, _name_to_pathes

    if key in _review_data:
        return _review_data[key]['path']

    if key in _name_to_pathes:
        if len(_name_to_pathes[key]) == 1:
            return _name_to_pathes[key][0]
        else:
            raise RuntimeError(
                'Please specify either hash key or path of %s.' % key)

    if key in _path_to_digest:
        return key

    raise RuntimeError('Do not know how to identify %s.' % key)

def load_raw_reviews(path):
    """
    Assume path points to a pickle file and has following structures:
    [review 0, review 1, ...], where a review is a list. And
    username is the 2nd element of the first element in each review;
    comment is the 4th element in each review;
    rating star is the 5th element in each review;
    timestamp is the 28th element in each review.
    """
    path = review_data_identifier(path)

    with open(path, 'rb') as fd:
        reviews = pickle.load(fd)

    if not np.iterable(reviews):
        raise ValueError

    R = np.empty(len(reviews), dtype=np.dtype([
            ('timestamp', np.int64),  # 8 bytes
            ('star', 'uint8'),        # 1 bytes
            ('username', 'U64'),      # 2(unicode) * 64(length) bytes
            ('review', object)        # varied
        ]))

    for n, review in enumerate(reviews):
        message = review[3]
        R[n]['timestamp'] = review[27]
        R[n]['star'] = review[4]
        R[n]['username'] = review[0][1]
        R[n]['review'] = '' if message is None else message
    I = np.argsort(R['timestamp'])

    return R[I]

class Array(object):
    _columns = []
    _dtypes = []

    def __init__(self, n):
        self._allocate(n)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return '<Array contains %d rows>' % len(self._data)

    def __getitem__(self, key):
        if not isinstance(key, slice) and \
           not np.shape(key) == () and \
           not (isinstance(key, np.ndarray) and key.dtype == np.bool_):
            key = np.unique(key)

        other = copy.copy(self)
        other._data = self._data.iloc[key]
        return other

    def __getstate__(self):
        state = {i: self._data[i].to_numpy() for i in self._columns}
        state['index'] = self.index
        return state

    def __setstate__(self, state):
        n = len(state['index'])
        self.__init__(n)
        self._data.index = state['index']

        for i in self._columns:
            self._data[i] = state[i]

    def _allocate(self, n):
        dt = np.dtype(list(zip(self._columns, self._dtypes)))
        self._data = pd.DataFrame(np.empty(n, dtype=dt))

    @property
    def index(self):
        return self._data.index.to_numpy()

    def loc(self, key):
        if not isinstance(key, slice) and \
           not np.shape(key) == () and \
           not (isinstance(key, np.ndarray) and key.dtype == np.bool_):
            key = np.unique(key)

        other = copy.copy(self)
        other._data = self._data.loc[key]
        return other

class TimeStamps(Array):
    _columns = Array._columns + ['timestamp']
    _dtypes = Array._dtypes + [np.int64]

    def __init__(self, n):
        super(TimeStamps, self).__init__(n)
        self._data['timestamp'] = 0

    def __repr__(self):
        return '<TimeStamps contains %d rows>' % len(self._data)

    @property
    def datetime(self):
        return pd.to_datetime(self.timestamps, unit='ms')

    @property
    def timestamps(self):
        return self._data['timestamp'].to_numpy()

    @timestamps.setter
    def timestamps(self, value):
        self._data['timestamp'] = value

    @property
    def years(self):
        return self.datetime.year.to_numpy()

    @property
    def months(self):
        return self.datetime.month.to_numpy()

    @property
    def days(self):
        return self.datetime.day.to_numpy()

    @property
    def hours(self):
        return self.datetime.hour.to_numpy()

    @property
    def minutes(self):
        return self.datetime.minute.to_numpy()

    @property
    def seconds(self):
        return self.datetime.second.to_numpy()

    @property
    def milliseconds(self):
        ms = self.datetime.microsecond.to_numpy() / 1000
        return ms.astype(np.int64)

    @property
    def microseconds(self):
        return self.datetime.microsecond.to_numpy()

    def strftime(self, format_code='%b %d, %Y'):
        """
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
        """
        return self.datetime.strftime(format_code).to_numpy()

class GoogleReviews(TimeStamps):
    _columns = TimeStamps._columns + ['username', 'star', 'review']
    _dtypes = TimeStamps._dtypes + ['U64', 'uint8', object]

    def __init__(self, path, discard_duplicates=True):
        path = review_data_identifier(path)
        data = load_raw_reviews(path)

        if discard_duplicates:
            M = np.r_[True, data[1:] != data[:-1]]
            data = data[M]

        super(GoogleReviews, self).__init__(len(data))

        self._discard_duplicates = discard_duplicates
        self._data['username'] = data['username']
        self._data['star'] = data['star']
        self._data['review'] = data['review']
        self._data['timestamp'] = data['timestamp']

        global _path_to_digest, _review_data
        digest = _path_to_digest[path]

        self._keys = [digest]
        self._name = _review_data[digest]['name']
        self._paths = [_review_data[digest]['path']]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return '<GoogleReviews %s contains %d rows>' % \
            (self._name, len(self._data))

    def __getstate__(self):
        state = {i: self._data[i].to_numpy() for i in self._columns}
        state['index'] = self.index
        state['name'] = self._name
        state['keys'] = self._keys
        state['discard_duplicates'] = self._discard_duplicates
        return state

    def __setstate__(self, state):
        discard_duplicates = state['discard_duplicates']
        key = state['keys'][0]
        self.__init__(key, discard_duplicates)

        n = len(state['index'])
        self._allocate(n)
        self._data.index = state['index']
        for i in self._columns:
            self._data[i] = state[i]

        global _review_data

        self._keys = state['keys']
        self._name = _review_data[key]['name']
        self._paths = [_review_data[i]['path'] for i in state['keys']]

    @property
    def name(self):
        return self._name

    @property
    def reviews(self):
        return self._data['review'].to_numpy()

    @property
    def usernames(self):
        return self._data['username'].to_numpy()

    @property
    def stars(self):
        return self._data['star'].to_numpy()

    def intersection(self, other):
        # assume each review has unique timestamps
        if not isinstance(other, GoogleReviews) or \
           not self._name == other.name:
            raise ValueError()

        T = np.intersect1d(self.timestamps, other.timestamps)
        I = np.searchsorted(self.timestamps, T)
        J = np.searchsorted(other.timestamps, T)
        assert(np.all(self.timestamps[I] == T))
        assert(np.all(other.timestamps[J] == T))

        M = (self.usernames[I] == other.usernames[J]) & \
            (self.reviews[I] == other.reviews[J]) & \
            (self.stars[I] == other.stars[J])
        I = I[M]

        return self[I]

    def diff(self, other):
        A = self.intersection(other)
        M = ~np.isin(self.index, A.index)
        return self[M]

    def union(self, other):
        A = other.diff(self)
        data = pd.concat([self._data, A._data])
        I = np.argsort(data['timestamp'].to_numpy())
        data = data.iloc[I]

        R = copy.copy(self)
        R._allocate(len(data))
        for i in self._columns:
            R._data[i] = data[i].to_numpy()

        R._keys = self._keys + other._keys
        R._paths = self._paths + other._paths
        return R
