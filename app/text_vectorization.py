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


def generate_data(descriptions, features, tokenizer, max_length, vocab_size):
    while 1:
        for key, desc_list in descriptions.items():
            feature = features[key][0]

            input_img, input_seq, output_seq = create_sequences(tokenizer, max_length, desc_list, feature, vocab_size)
            yield [[input_img, input_seq], output_seq]


def create_sequences(tokenizer, max_length, desc_list, feature, vocab_size):
    photo_for_seq, input_sequences, output_for_seq = list(), list(), list()
    # walk through each image identifier
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
            photo_for_seq.append(feature)
            input_sequences.append(in_seq)
            output_for_seq.append(out_seq)
    return array(photo_for_seq), array(input_sequences), array(output_for_seq)


def size_of_vocab(tokenizer):
    return len(tokenizer.word_index) + 1


def create_tokenizer(description_list):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(description_list)
    return tokenizer

def get_reverse_tokenizer(tokenizer: Tokenizer):
    return {v:k for (k,v) in tokenizer.word_index.items()}


def flatten_descriptions(descriptions):
    return [item for sub in descriptions.values() for item in sub]


if __name__ == "__main__":
    filename = 'data/info_on_images/Flickr_8k.trainImages.txt'
    descriptions = get_clean_descriptions(load_file(filename), 'utils/descriptions.txt')

    flatten = flatten_descriptions(descriptions)

    tokenizer = create_tokenizer(flatten)
    features = get_feature_vector(descriptions.keys(), 'utils/features.pkl')
    max_length = max_len([i.split() for i in flatten])
    size = len(to_bag_of_words(descriptions))
