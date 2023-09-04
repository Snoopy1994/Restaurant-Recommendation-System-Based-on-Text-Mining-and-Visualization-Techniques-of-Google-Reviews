import re
import emoji
from ckiptagger import construct_dictionary, WS, POS, NER

_is_set_data = False

def set_CKIP_data(data_path):
    """loading data"""

    global CKIP_WS, CKIP_POS, CKIP_NER, _is_set_data
    CKIP_WS = WS(data_path)    # word segmentation
    CKIP_POS = POS(data_path)  # part-of-speech tagging
    CKIP_NER = NER(data_path)  # named entity recognition
    _is_set_data = True

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

def apply_CKIP(comments, dictionary=[]):
    if not isinstance(comments, list):
        raise ValueError('only accept a list of string')

    if isinstance(dictionary, dict):
        dictionary = construct_dictionary(dictionary)

    for c in comments:
        if not isinstance(c, str):
            raise ValueError('only accept a list of string')

    if not _is_set_data:
        raise ValueError(
            "CKIP model is not set yet. Use set_CKIP_data at first.")

    # segment words by CKIP
    word_sentence_list = CKIP_WS(
        comments, recommend_dictionary=dictionary)
    pos_sentence_list = CKIP_POS(
        word_sentence_list)
    entity_sentence_list = CKIP_NER(
        word_sentence_list, pos_sentence_list)

    result = {
        'ws': word_sentence_list,
        'pos': pos_sentence_list,
        'ner': entity_sentence_list
    }

    return result
