# Примеры программ для курса "Программирование глубоких нейронных сетей на Python"

[Страница курса с видеолекциями и практическими заданиями](https://www.asozykin.ru/courses/nnpython).

## Примеры

1. Распознавание рукописных цифр из набора данных [MNIST](http://yann.lecun.com/exdb/mnist/) - `mnist`. Используется полносвязная и сверточная нейронные сети.
2. Распознавание объектов на изображениях из набора данных [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - `cifar10`. Используется сверточная нейронная сеть.
3. Определение тональности отзывов на фильмы из [IMDB Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) - `imdb`. Используется рекуррентная сеть LSTM.
4. Прогноз стоимости домов для набора данных [Boston Housing](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) - `regression`. Пример решения задачи регрессии.
5. Использование предварительно обученных нейронных сетей - `pretrained_networks`
6. Сохранение обученной нейронной сети - `saving_models`.
7. Примеры задач компьютерного зрения - `computer_vision`.

## Необходимое ПО

1. Python 3.
2. Библиотека глубокого обучения [Keras](https://keras.io/).
3. Библиотеки  [TensorFlow](https://www.tensorflow.org/) или [Theano](http://deeplearning.net/software/theano/) (используются в качестве вычислительного бекенда для Keras).

Инструкция по установке:

- [Keras и TensorFlow в Anaconda](https://www.asozykin.ru/deep_learning/2017/09/07/Keras-Installation-TensorFlow.html).
- [Keras и Theano в Anaconda](https://www.asozykin.ru/deep_learning/2016/12/25/Keras-Installation.html).

Примеры тестировались с TensorFlow. При использовании Theano возможны проблемы из-за разных подходов к хранению изображений.

## Благодарности

При реализации проекта используются средства поддержки, выделенные в качестве гранта на основании конкурса, проведенного Общероссийской общественно-государственной просветительской организации «Российское общество «Знание».
