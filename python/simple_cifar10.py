from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

# adapted from:
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

def basic_conv(input_shape, nb_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Convolution2D(96, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# general configuration
batch_size = 64
nb_classes = 10
nb_epoch = 100

# squint at this, and it's destructuring
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

traingen = ImageDataGenerator(
    rotation_range=15, # degrees
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
traingen.fit(X_train)

model = basic_conv(X_train.shape[1:], nb_classes)
model.fit_generator(traingen.flow(X_train, Y_train,
                                  batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(X_test, Y_test)
)

with open("simple.json", "w") as f:
    f.write(model.to_json())
model.save_weights("simple.h5")
