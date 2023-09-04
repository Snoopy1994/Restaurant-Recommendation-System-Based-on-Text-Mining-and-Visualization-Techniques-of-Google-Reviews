import os
import re
import numpy as np
import pandas as pd

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.isdir(path):
        raise ValueError('Wrong input:\n%s should be a directory of folder' % path)

def find_duplicates(elements):
    """
    Input
    -----
    `elements`: list of str

    Output
    ------
    list of duplicate str in `elements`,
    notice that N duplicate same element only show once in output
    """
    A = set()
    B = set()
    for i in elements:
        if i in A:
            B.add(i)
        else:
            A.add(i)
    return list(B)

def load_dictionary(path, encoding='utf-8'):
    with open(path, "r", encoding=encoding) as fd:
        A = fd.read().splitlines()

    A = [i.strip().split(" ") for i in A]
    A = [i for i in A if len(i) == 2]

    keys = find_duplicates([i for i, j in A])
    if len(keys) > 0:
        raise RuntimeError(
            "{keys}\n are duplicate, please check {path}!".format(keys=keys, path=path))

    data = {}
    for i, j in A:
        try:
            data[i] = float(j)
        except:
            print("%s has wrong weight %s" % (i, j))

    return data

def label_change(labels):
    """
    Parameters
    ----------
    labels : 1-D array-like
             a label can be bool, number, word

    Returns:
    indices : array-like
              positions of labels where label change
    """
    labels = np.array(labels)
    if len(labels.shape) != 1:
        raise ValueError('labels should be 1-D array.')

    diff = (labels[:-1] != labels[1:])
    I = np.where(diff)[0] + 1
    return I

def sliding_sum(labels, half_window_size):
    """
    summarize a window of nearby values

    Parameters
    ----------
    labels: 1-D array-like
            label is number

    Returns
    -------
    sum: 1-D array-like
         shape of sum is same as labels

    Notes
    -----
    step by step of this implementation
        1. labels = [A, B, C, D]
           WSize(half_window_size) = 2

        2. X => [A, B, C, D], shape=(N,)

        3. repeat the first and the last elements
           X => [A, A, A, B, C, D, D, D], shape=(N+2*WSize,)

        4. cumlative sum
           X => [0, A, A+A, ..., A+A+...+D], shape=(N+2*WSize+1,)

        5. Thus, the first value is the difference of (0, 2*WSize+1),
           and the last one is difference of (N-1, N+2*WSize)
    """
    N = 2 * half_window_size + 1

    X = np.array(labels).astype(np.float64)
    X = np.r_[
        np.repeat(X[0], half_window_size), X,
        np.repeat(X[-1], half_window_size)]

    X = np.r_[0, np.cumsum(X)]
    return X[N:] - X[:-N]

def sliding_mean(labels, half_window_size):
    N = 2 * half_window_size + 1
    X = sliding_sum(labels, half_window_size) / N
    return X

def file_directory_DFS(root_path, condition):
    assert(callable(condition))

    R = set()

    def f(path):
        if condition(path):
            R.add(path)

        if not os.path.isdir(path):
            return

        for i in os.listdir(path):
            f(os.path.join(path, i))

    f(root_path)
    return R

def discover_pickles(path):
    def f(x):
        return os.path.split(x)[-1][-4:] == '.pkl'

    return file_directory_DFS(path, f)

def match_string_segments(base_string, segments):
    assert(isinstance(base_string, str))
    assert(np.iterable(segments))
    for s in segments:
        assert(isinstance(s, str))

    def f(A, B):
        r = re.search(B, A)
        if r is None:
            return (0, 0)
        else:
            return (r.start(), r.end())

    indices = []
    current_index = 0
    for segment in segments:
        begin, end = f(base_string, segment)
        base_string = base_string[end:]

        indices.append((begin + current_index, end + current_index))
        current_index += end
    return indices

def batch_substring_finding(strings, pattern):
    # define a shorthand for find substring in an array
    return pd.Series(strings).str.find(pattern).to_numpy() == 0
