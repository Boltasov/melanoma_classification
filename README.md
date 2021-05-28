# melanoma_classification
project for ML/DL courses

Для решения задачи воспроизведён код из статьи “Свёрточные нейронные сети для классификации изображений меланомы”.

Ссылка на эту статью на paperswithcode: https://paperswithcode.com/paper/convolutional-neural-networks-for-classifying
Ссылка на исходный репозиторий: https://github.com/abhinavsagar/skin-cancer

Данные возяты из открытой базы данных ISIC. По этим ссылкам они доступны в виде архивов, что ускоряет скачивание:
- Training data: https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip
- Validation data: https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip
- Test data: https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip

Для воспроизведения кода подразумевается, что архивы будут распакованы в той же директоии, где находится файл classification.ipynb

## Шаги работы над моделью:

1) Делим данные на тестовую и обучаемую части
2) Делаем аугментации: обрезка, масштабирование, отражение и изменение яркости, чтобы увеличить размер набора данных.
3) Используем предобученную модель ResNet 50 (сверточная нейронная сеть) и делаем тонкую настройку последних нескольких слоёв.
4) Используем  50 % исключение (dropout) и пакетную нормализацию (batch normalization) для уменьшения переобучения
5) Используем два полносвязных слоя с 64 нейронами и 2 нейронами соответственно. Последний слой используется для классификации с softmax в качестве функции активации.
6) Используем binary cross entropy в качестве loss функции
7) Обучим модель для 50 эпох с размером пакета 64, выбрав скорость обучения 0,0001, оптимизатор Adam.

## Комментарии к репозиторию:

classification.ipynb - обучение модели и сохранение весов в файл weights.best.hdf5 (воспроизведён из исходного репозитория)
keras_model - воспроизведение модели на полученных весах
config.py - сюда нужно положить токен чат-бота
photo_conversion.py - преобразует фотографии в формат, подходящий для модели
main.py - обеспечивает обработку запросов чат-ботом

Ссылка на презентацию, которую я показывал при сдаче: https://drive.google.com/file/d/1jNJ967kQzdFqvr50rkxWOWkN5y64HrKl/view?usp=sharing
