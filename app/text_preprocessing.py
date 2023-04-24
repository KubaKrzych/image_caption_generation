import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

STOP_WORDS = set(stopwords.words('english'))
BEGIN = '^'
END = '$'
SEP = '###'
STEMMER = PorterStemmer()


def is_stop_token(text: str):
    global BEGIN, END
    return text == BEGIN or text == END


def text_preprocessing(text: str):
    global STOP_WORDS, BEGIN, END, STEMMER
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # START - text - END
    text = BEGIN + ' ' + text + ' ' + END
    # THE BELOW ARE NOT NECESSARY AND MIGHT BE TROUBLESOME
    text = text.split()
    # text = list(filter(lambda x: (x not in STOP_WORDS or is_stop_token(x)) and not x.isnumeric(), text))
    text = " ".join(text)

    return text


def load_file(path: str):
    with open(path, 'r') as file:
        text = file.read().splitlines()
    return text


def get_descriptions(text):
    descriptions = {}

    for line in text:
        id, desc = re.split('#\d\s+',line)
        if id not in descriptions:
            descriptions[id] = []
        descriptions[id].append(text_preprocessing(desc))
    return descriptions


def to_bag_of_words(descriptions):
    words = set()
    for key in descriptions.keys():
        [words.update(w.split(' ')) for w in descriptions[key]]
    return words


def save_descriptions_to_a_file(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + SEP + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


if __name__ == "__main__":
    filename = 'data/info_on_images/Flickr8k.token.txt'
    doc = load_file(filename)
    # parse descriptions
    descriptions = get_descriptions(doc)
    print('Loaded: %d ' % len(descriptions))
    # summarize vocabulary
    vocabulary = to_bag_of_words(descriptions)
    print('Vocabulary Size: %d' % len(vocabulary))
    # save to file
    save_descriptions_to_a_file(descriptions, 'utils/descriptions.txt')