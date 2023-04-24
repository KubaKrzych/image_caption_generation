from keras.layers import Input, Dropout, Dense, Embedding, LSTM, add
from keras.models import Model
from keras.utils import plot_model
from text_vectorization import size_of_vocab, create_tokenizer, flatten_descriptions, max_len, generate_data
from load_data import load_file, get_clean_descriptions, get_feature_vector
from pickle import dump

def create_model(vocab_size, max_length_of_seq):
    image_input = Input(shape=(4096,))
    image = Dropout(0.5)(image_input)
    image = Dense(256, activation='relu')(image)

    sequences_input = Input(shape=(max_length_of_seq,))
    sequences = Embedding(vocab_size, 256, mask_zero=True)(sequences_input)
    sequences = Dropout(0.5)(sequences)
    sequences = LSTM(256)(sequences)

    decoder = add([image, sequences])
    decoder = Dense(256, activation='relu')(decoder)
    output = Dense(vocab_size, activation='softmax')(decoder)

    model = Model(inputs=[image_input, sequences_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print(model.summary())
    plot_model(model, to_file='utils/model.png', show_shapes=True)

    return model


def train_model(file_path, epochs):
    train_images = load_file(file_path)
    features = get_feature_vector(train_images, "utils/features.pkl")
    descriptions = get_clean_descriptions(train_images, "utils/descriptions.txt")
    flatten_description = flatten_descriptions(descriptions)
    tokenizer = create_tokenizer(flatten_description)

    vocab_size = size_of_vocab(tokenizer)
    max_length = 34 #TO JEST ZLE
    model = create_model(vocab_size, max_length)

    steps = len(flatten_description)
    for i in range(epochs):
        gen = generate_data(descriptions, features, tokenizer, max_length, vocab_size)
        model.fit(gen, epochs=1, steps_per_epoch=steps,verbose=1)
        model.save('models/model_unstemmed_' + str(i) + '.h5')


if __name__ == "__main__":
    train_model('data/info_on_images/Flickr_8k.trainImages.txt', 5)
