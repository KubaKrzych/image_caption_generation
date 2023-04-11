import tensorflow as tf
import pickle

from load_data import get_clean_descriptions, load_file
from keras.preprocessing.text import Tokenizer

def max_len(flatten_descriptions):
    return max((len(l) for l in flatten_descriptions))


def rnn_data_gen():
    return None

def create_sequences_of_words():
    

filename = 'data/info_on_images/Flickr_8k.trainImages.txt'
descriptions = get_clean_descriptions(load_file(filename), 'utils/descriptions.txt')

flatten = [item for sub in descriptions.values() for item in sub]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(flatten)
pickle.dump(tokenizer, open('utils/vectorized_vocab.pkl', 'wb'))

