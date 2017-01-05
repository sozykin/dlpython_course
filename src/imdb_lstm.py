import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# Устанавливаем seed для повторяемости результатов
np.random.seed(42)
# Максимальное количество слов (по частоте использования)
max_features = 5000
# Максимальная длина рецензии в словах
maxlen = 80

# Загружаем данные
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)

# Заполняем или обрезаем рецензии
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# Создаем сеть
model = Sequential()
# Слой для векторного представления слов
model.add(Embedding(max_features, 32, dropout=0.2))
# Слой долго-краткосрочной памяти
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
# Полносвязный слой
model.add(Dense(1, activation="sigmoid"))

# Копмилируем модель
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Обучаем модель
model.fit(X_train, y_train, batch_size=64, nb_epoch=7,
          validation_data=(X_test, y_test), verbose=1)
# Проверяем качество обучения на тестовых данных
scores = model.evaluate(X_test, y_test,
                        batch_size=64)
print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))
