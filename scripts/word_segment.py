import sys
sys.path.append('..')

import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import emoji
import pickle
import argparse

import numpy as np

# from alive_progress import alive_bar
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

import utils
# import word_segment as ws
import google_reviews

input_folder = ''
output_folder = ''
_data = "/Users/nadia/CKIP/data/"
_recommendation = "/Users/nadia/Desktop/Projects/recommendation_dic.txt"

#load model
word_segmentation = WS(_data) # word segmentation
pos_tagging = POS(_data) # part-of-speech tagging
ner = NER(_data) #named entity recognition

def load_pkl(path):
    with open(path, 'rb') as fd:
        return pickle.load(fd)

def save_pkl(path, obj):
    with open(path, 'wb') as fd:
        return pickle.dump(obj, fd)

def apply_CKIP(comments, dictionary=[]):
    if not isinstance(comments, list):
        raise ValueError('only accept a list of string')

    for c in comments:
        if not isinstance(c, str):
            raise ValueError('only accept a list of string')

    # segment words by CKIP
    word_sentence_list = word_segmentation(comments, recommend_dictionary = dictionary)
    pos_sentence_list = pos_tagging(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)

    result = {
        'ws': word_sentence_list,
        'pos': pos_sentence_list,
        'ner': entity_sentence_list
    }

    return result

def word_segment_preprocess(comment):
    if not isinstance(comment, str):
        raise ValueError('comment must be a string.')

    # substitute emoji to space
    comment = emoji.replace_emoji(comment, ' ')

    # substitute punctuations to spaces
    comment = re.sub(r'[^\w\s]',' ', comment)

    # remove heading/trailing spaces and dublicate spaces
    comment = " ".join(comment.split())
    return comment

def main(use_word_weights=False):

    utils.mkdir(output_folder)
    print(input_folder)
    print(output_folder)

    if not os.path.exists(input_folder):
        raise ValueError()

    if not os.path.exists(output_folder):
        raise ValueError()

    path = os.path.join(input_folder, '*.pkl')
    files = glob.glob(path)

    # load recommend dictionary
    dictionary = []
    if use_word_weights:
        print("Word weight is used.")
        word_to_weight = utils.load_dictionary(_recommendation)
        dictionary = construct_dictionary(word_to_weight)


    # main works
    # with alive_bar(len(files), force_tty=True) as bar:

    for file in files:

        name = os.path.split(file)[-1]
        if name in os.listdir(output_folder):
            # bar()
            continue

        print('process %s ...' % file)

        data = load_pkl(file)
        mask = utils.batch_substring_finding(data.reviews, "(由 Google 提供翻譯)")
        data = data[~mask]

        comments = [word_segment_preprocess(i) for i in data.reviews]
        comments = np.array(comments)

        mask = comments == ""
        data = data[~mask]
        comments = comments[~mask]
        comments = comments.tolist()

        result = apply_CKIP(comments, dictionary=dictionary)
        result['data'] = data

        f = os.path.join(output_folder, os.path.split(file)[-1])
        save_pkl(f, result)

            # bar()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input folder")
    parser.add_argument("--output", help="output folder")
    parser.add_argument("--use_word_weights", action='store_true', help="using recommend_dictionary")

    args = parser.parse_args()

    # global input_folder, output_folder
    if args.input is not None:
        input_folder = args.input

    if args.output is not None:
        output_folder = args.output

    main(use_word_weights=args.use_word_weights)