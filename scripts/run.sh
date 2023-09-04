#!/bin/sh
python word_segment.py --input ~/Desktop/Projects/cleaned_data/taichung --output ~/Desktop/Projects/cleaned_data/taichung/word_segments --use_word_weights
python word_segment.py --input ~/Desktop/Projects/cleaned_data/taipei --output ~/Desktop/Projects/cleaned_data/taipei/word_segments --use_word_weights
python word_segment.py --input ~/Desktop/Projects/cleaned_data/tainan --output ~/Desktop/Projects/cleaned_data/tainan/word_segments --use_word_weights