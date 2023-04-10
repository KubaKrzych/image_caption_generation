import tensorflow as tf
import os

from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils.image_utils import load_img, img_to_array
from keras.models import Model
from pickle import dump

SIZE = (224, 224)

if __name__ == "__main__":
        
    path = os.path.join("data", "info_on_images", "Flickr_8k.trainImages.txt")
    model = VGG16()
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    print(model.summary())

    features_vectors = {}

    with open(path, "r") as file:
        data = file.read().splitlines()

    for image_name in data:
        image = load_img(os.path.join("data", "fetched", image_name), 
                        target_size=SIZE)
        image = img_to_array(image)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

        image = preprocess_input(image)

        feature = model.predict(image, verbose=2)

        features_vectors[image_name] = feature
    
    dump(features_vectors, open(os.path.join("utils", "features.pkl"), "wb"))