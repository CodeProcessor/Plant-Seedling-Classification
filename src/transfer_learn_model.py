'''
Created on 9/1/20

@author: dulanj
'''
import tensorflow as tf

from src.model import Model


class TransferLearnModel(Model):
    def __init__(self):
        pass

    @staticmethod
    def get_model(verbose):
        model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(80, 80, 3))
        input_layer = model.inputs
        x = model.layers[-1].output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(32, activation='softmax')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(12, activation='softmax')(x)
        model = tf.keras.Model(input_layer, x)
        if verbose == 1:
            print(model.summary())

        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


if __name__ == "__main__":
    obj = TransferLearnModel()
    obj.get_model()
