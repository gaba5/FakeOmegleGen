from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import os
import io
from prepare_data import prep_data
import joblib


prep_data()
path = "prepared_data.txt"
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 50
step = 5
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2, 1.5, 2.0]:
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        for i in range(1000):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated+=next_char
        file_path=f"training_generated_conversations/Epoch-{epoch}"
        if not os.path.exists("training_generated_conversations"):
            os.makedirs("training_generated_conversations")
        if not os.path.exists(f"training_generated_conversations/Epoch-{epoch}"):
            os.makedirs(f"training_generated_conversations/Epoch-{epoch}")

        with open(file_path+f"/{diversity}.txt","w") as f:
            f.write(generated)


def gen_text(length,output_name):
    diversity=1.2
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    for i in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        sentence = sentence[1:] + next_char
        generated += next_char
    with open(output_name, "w") as f:
        f.write(generated)


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
epochs=60
if not os.path.exists('omegle_convo_maker.h5'):
    model.fit(x, y,
              batch_size=128,
              epochs=epochs,
              callbacks=[print_callback])
    joblib.dump(model,"omegle_convo_maker.h5")
else:
    model=joblib.load("omegle_convo_maker.h5")
    gen_text(5000,"generated.txt")