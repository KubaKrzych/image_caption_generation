import pickle

from text_preprocessing import load_file, SEP


def get_clean_descriptions(image_ids, descriptions_path):
    descriptions = load_file(descriptions_path)
    descriptions = list(map(lambda x: x.split(SEP), descriptions))
    multiple_descriptions = {}
    for k,v in descriptions:
        if k not in image_ids:
            continue
        elif k not in multiple_descriptions:
            multiple_descriptions[k] = []
        multiple_descriptions[k].append(v)

    return multiple_descriptions


def get_feature_vector(image_ids, features_path):
    features_vectors = pickle.load(open(features_path, 'rb'))
    return {k:v for (k,v) in features_vectors.items() if k in image_ids}


if __name__ == "__main__":
    filename = 'data/info_on_images/Flickr_8k.trainImages.txt'
    train = load_file(filename)
    print('Dataset: %d' % len(train))
    train_descriptions = get_clean_descriptions(train, 'utils/descriptions.txt')
    print('Descriptions: train=%d' % len(train_descriptions))
    train_features = get_feature_vector(train, 'utils/features.pkl')
    print('Photos: train=%d' % len(train_features))
    print(train_descriptions.items())

