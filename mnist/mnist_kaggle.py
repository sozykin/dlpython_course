import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import sys
import os.path

# Устанавливаем seed для повторяемости результатов
np.random.seed(42)

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
OUTPUT_FILE = "submission.csv"

# Размер изображения
img_rows, img_cols = 28, 28

if not os.path.isfile(TRAIN_FILE) or not os.path.isfile(TEST_FILE):
    print("""Загрузите с kaggle данные для обучения и тестирования
             (файлы train.csv и test.csv) и запишите их в текущий каталог
          """)
    sys.exit()


# Загружаем данные для обучения
train_dataset = np.loadtxt(TRAIN_FILE, skiprows=1, dtype='int', delimiter=",")
# Выделяем данные для обучения
x_train = train_dataset[:, 1:]
# Переформатируем данные в 2D, бэкенд Tensorflow
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
# Нормализуем данные
x_train = x_train.astype("float32")
x_train /= 255.0
# Выделяем правильные ответы
y_train = train_dataset[:, 0]
# Преобразуем правильные ответы в категоризированное представление
y_train = np_utils.to_categorical(y_train)



#Загружаем данные для предсказания
test_dataset = np.loadtxt(TEST_FILE, skiprows=1, delimiter=",")
# Переформатируем данные в 2D, бэкенд TensorFlow
x_test = test_dataset.reshape(test_dataset.shape[0], img_rows, img_cols, 1)
x_test /= 255.0


# Создаем последовательную модель
model = Sequential()

model.add(Conv2D(75, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(100, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# Компилируем модель
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

# Обучаем сеть
model.fit(x_train, y_train, batch_size=200, epochs=10, verbose=2)

# Making predictions
predictions = model.predict(x_test)
# Converting from categorical to classed
predictions = np.argmax(predictions, axis=1)

# Saving predictions to the file
out = np.column_stack((range(1, predictions.shape[0]+1), predictions))
np.savetxt(OUTPUT_FILE, out, header="ImageId,Label", comments="", fmt="%d,%d")
