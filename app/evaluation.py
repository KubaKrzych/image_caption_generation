from pickle import load
from keras.models import Model
from text_preprocessing import BEGIN, END, get_descriptions, load_file
from text_vectorization import get_reverse_tokenizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from keras.applications import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.utils import load_img, img_to_array
from photo_preprocessing import SIZE
from keras.models import load_model
from numpy import argmax

def predict_description(model, tokenizer: Tokenizer, photo, max_length):
    input = BEGIN
    reversed_tokenizer = get_reverse_tokenizer(tokenizer)
    for i in range(max_length):
        
        sequence = tokenizer.texts_to_sequences([input])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_predict = model.predict([photo, sequence], verbose=0)
        y_predict = argmax(y_predict)

        predicted_word: str = reversed_tokenizer[y_predict]

        if predicted_word == None:
            break

        input += " " + predicted_word

        if predicted_word == END:
            break
    return input

def get_clean_descriptions(filename, photo_ids):
    descs = get_descriptions(load_file(filename))
    return {k:v for (k,v) in descs.items() if k in photo_ids}

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    y, y_pred = [], []
    for k, desc_list in descriptions.items():
        predicted_description = predict_description(model, tokenizer, photos[k], max_length)
        refs = [desc.split() for desc in desc_list]
        y.append(refs)
        y_pred.append(predicted_description)

    print('BLEU-1: %f' % corpus_bleu(y, y_pred, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(y, y_pred, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(y, y_pred, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(y, y_pred, weights=(0.25, 0.25, 0.25, 0.25)))


def get_photo_ids(filename):
    text = load_file(filename)
    return [i.split('.')[0] for i in text]

if __name__ == "__main__":
    model = load_model('models/model_0.h5')
    ids_filename = 'data/info_on_images/Flickr_8k.testImages.txt'

    ids = get_photo_ids(ids_filename)
