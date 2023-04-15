import tensorflow as tf
import pickle

from load_data import get_clean_descriptions, load_file
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from numpy import array
from load_data import get_feature_vector
from text_preprocessing import to_bag_of_words

def max_len(flatten_descriptions):
    return max((len(l) for l in flatten_descriptions))


def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    photo_for_seq, input_sequences, output_for_seq = list(), list(), list()
    # walk through each image identifier
    for key, desc_list in descriptions.items():
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                photo_for_seq.append(photos[key][0])
                input_sequences.append(in_seq)
                output_for_seq.append(out_seq)
    return array(photo_for_seq), array(input_sequences), array(output_for_seq)


filename = 'data/info_on_images/Flickr_8k.trainImages.txt'
descriptions = get_clean_descriptions(load_file(filename), 'utils/descriptions.txt')
flatten = [item for sub in descriptions.values() for item in sub]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(flatten)

create_sequences(
    tokenizer, max_len(flatten), descriptions, get_feature_vector(descriptions.keys(), 'utils/features.pkl'), len(to_bag_of_words(descriptions))+1)
# pickle.dump(tokenizer, open('utils/vectorized_vocab.pkl', 'wb'))
