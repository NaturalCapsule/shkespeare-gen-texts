import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode().lower()

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))

index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGHT = 40
STEP_SIZE = 3

sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGHT, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGHT])
    next_characters.append(text[i + SEQ_LENGHT])

x = np.zeros((len(sentences), SEQ_LENGHT, len(characters)), dtype = np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype = np.bool_)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

def rnn():
    model = Sequential(
        [
            LSTM(units = 128, input_shape = (SEQ_LENGHT, len(characters))),
            Dense(units = len(characters), activation = 'softmax')

        ]
    )
    return model

rnn = rnn()

rnn.compile(RMSprop(lr = 0.01), loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ['accuracy'])
rnn.fit(x = x, y = y, epochs = 4, batch_size = 256)


model = tf.keras.models.load_model('text.model')

def sample(preds, temp = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(lenght, temp):
    start_idx = random.randint(0, len(text) - SEQ_LENGHT - 1)
    generated_text = ''
    sentence = text[start_idx: start_idx + SEQ_LENGHT]
    generated_text += sentence
    for i in range(lenght):
        x = np.zeros((1, SEQ_LENGHT, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        predictions = model.predict(x, verbose = 0)[0]
        next_idx = sample(predictions, temp)
        next_character = index_to_char[next_idx]

        generated_text += next_character
        sentence = sentence[1:] + next_character

    return generated_text


print('-------------------------------------')
print(generate_text(300, 0.2))