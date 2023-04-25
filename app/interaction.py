from evaluation import *
from keras.models import load_model
from pickle import load
import sys

def extract_features(file):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    img = load_img(file, target_size=SIZE)
    img = img_to_array(img)
    img = img.reshape(1, img.shape[0],img.shape[1],img.shape[2])
    return model.predict(img, verbose=0)

if __name__ == "__main__":
    model = load_model('models/model_4.h5')
    path_to_img = sys.argv[1]
    
    tokenizer = load(open('utils/tokenizer.pkl', 'rb'))

    filename = 'data/info_on_images/Flickr_8k.testImages.txt'
    descriptions_path =  'data/info_on_images/Flickr8k.token.txt'
    ids = get_photo_ids(filename)
    descs = get_clean_descriptions(descriptions_path, ids)

    photo = extract_features(path_to_img)
    description = predict_description(model, tokenizer, photo, 34)
    print(description)
