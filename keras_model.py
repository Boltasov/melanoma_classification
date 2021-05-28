from keras import layers
from keras.models import Sequential
from keras import optimizers
from keras.applications.resnet import ResNet50
from keras import backend as K
import gc


def build_model(backbone, lr=5e-4):
    # Строим точно такую же архитектуру модели, как при обучении
    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.adam_v2.Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    return model


def init_model():
    # Инициализируем модель весами, полученными при обучении
    K.clear_session()
    gc.collect()

    resnet = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    model = build_model(resnet ,lr = 1e-4)
    model.summary()

    model.load_weights("weights.best.hdf5")
    return model


def predict(model,X):
    Y_pred = model.predict(X)
    return Y_pred
