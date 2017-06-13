from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# Создаем модель с архитектурой VGG16 и загружаем веса, обученные
# на наборе данных ImageNet
model = VGG16(weights='imagenet')

# Загружаем изображение для распознавания, преобразовываем его в массив
# numpy и выполняем предварительную обработку
img_path = 'ship.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Запускаем распознавание объекта на изображении
preds = model.predict(x)

# Печатаем три класса объекта с самой высокой вероятностью
print('Результаты распознавания:', decode_predictions(preds, top=3)[0])
