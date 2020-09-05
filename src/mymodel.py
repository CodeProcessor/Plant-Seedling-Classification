'''
Created on 9/5/20

@author: prislk
'''
import tensorflow as tf
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger


from src.model import Model


class MyModel(Model):
    def __init__(self):
        pass

    @staticmethod
    def get_model(verbose=1):
        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(80, 80, 3), activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization(axis=3))
        model.add(Dropout(0.1))

        model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization(axis=3))
        model.add(Dropout(0.1))

        model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization(axis=3))
        model.add(Dropout(0.1))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(12, activation='softmax'))

        model.summary()

        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if verbose == 1:
            print(model.summary())
        return model





if __name__ == "__main__":
    obj = MyModel()
    obj.model()