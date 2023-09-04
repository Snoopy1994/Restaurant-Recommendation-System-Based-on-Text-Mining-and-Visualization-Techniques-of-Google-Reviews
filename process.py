import os
import pickle
import argparse
import pandas as pd
import utils    

from alive_progress import alive_bar
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

input_folder = 'temporary_cache'
output_folder = 'output'
input_folder = os.path.join(os.getcwd(), input_folder)
output_folder = os.path.join(os.getcwd(), output_folder)

#load model
word_segment = WS("./data") # word segmentation
pos_tagging = POS("./data") # part-of-speech tagging
ner = NER("./data") #named entity recognition

def apply_CKIP(comments, dictionary=[]):
    if not isinstance(comments, list):
        raise ValueError('only accept a list of string')

    for c in comments:
        if not isinstance(c, str):
            raise ValueError('only accept a list of string')

    # segment words by CKIP
    word_sentence_list = word_segment(comments, recommend_dictionary = dictionary)
    pos_sentence_list = pos_tagging(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)

    result = {
        'ws': word_sentence_list,
        'pos': pos_sentence_list,
        'ner': entity_sentence_list
    }

    return result


def main(use_word_weights=False):
    utils.mkdir(input_folder)
    utils.mkdir(output_folder)
    print(input_folder)
    print(output_folder)

    files = os.listdir(input_folder)
    files = [i for i in files if i not in os.listdir(output_folder)]
    files = [i[:-4] for i in files if i[-4:] == '.csv']

    # load recommend dictionary
    dictionary = []
    if use_word_weights:
        print("Word weight is used.")
        word_to_weight = utils.load_dictionary('recommend_dictionary')
        dictionary = construct_dictionary(word_to_weight)


    # main works
    with alive_bar(len(files), force_tty=True) as bar:

        for file in files:

            # get the csv file
            data = pd.read_csv(os.path.join(input_folder, file + '.csv'))

            # transfer the file into output folder
            data.to_csv(os.path.join(output_folder, file + '.csv'))

            comments = data['comments'].to_list()

            result = apply_CKIP(comments, dictionary=dictionary)
            with open(os.path.join(output_folder, file + '.pkl'), 'wb') as f: 
                pickle.dump(result, f)

            bar()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input folder")
    parser.add_argument("--output", help="output folder")
    parser.add_argument("--use_word_weights", help="using recommend_dictionary")

    args = parser.parse_args()

    # global input_folder, output_folder
    if args.input is not None:
        input_folder = args.input

    if args.output is not None:
        output_folder = args.output

    use_word_weights = False
    if args.use_word_weights is not None:
        use_word_weights = True

    main(use_word_weights=use_word_weights)
