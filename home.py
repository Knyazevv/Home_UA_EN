import numpy as np

from keras.layers import Dense, LSTM, Input, Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras. preprocessing.sequence import pad_sequences


with open('train_data_ua', 'r', encoding='utf-8') as f:
    text_ua = f.readlines()
    text_ua[0] = text_ua[0].replace('\ufeff', '')

with open('train_data_en', 'r', encoding='utf-8') as f:
    text_en = f.readlines()
    text_en[0] = text_en[0].replace('\ufeff', '')


texts = text_ua + text_en
count_true = len(text_ua)
count_false = len(text_en)
total_lines = count_false + count_true
print(total_lines)


maxWords = 1000
tokenizer = Tokenizer(maxWords, lower=True, split=" ", char_level=False)
tokenizer.fit_on_texts(texts)

dist = list(tokenizer.word_counts.items())
print(dist[:10])
print(texts[0][:100])


max_text_len = 10
data = tokenizer.texts_to_sequences(texts)
data_pad = pad_sequences(data, max_text_len)
print(data_pad)


X = data_pad
Y = np.array([[1, 0]]*count_true + [[0, 1]]*count_false)
print(X.shape, Y.shape)


indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]


model = Sequential()
model.add(Embedding(maxWords, 128, input_length=max_text_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(2, activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy', metrics=[
              'accuracy'], optimizer=Adam(0.001))
history = model.fit(X, Y, batch_size=32, epochs=50)


reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


def sequence_to_text(list_of_indeces):
    words = [reverse_word_map.get(letter) for letter in list_of_indeces]
    return (words)


t = "Live to see yourself become stronger and live a happier, fuller life."
data = tokenizer.texts_to_sequences([t])
data_pad = pad_sequences(data, maxlen=max_text_len)
print(sequence_to_text(data_pad))

res = model.predict(data_pad)
print(res, np.argmax, sep='\n')
