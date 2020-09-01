'''
Created on 9/1/20

@author: dulanj
'''
import tensorflow as tf
import keras.layers as all_layers

from src.model import Model


class TransferLearnModel(Model):
    def __init__(self):
        pass

    @staticmethod
    def get_model(verbose):
        model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        input_layer = model.inputs
        x = model.layers[-1].output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(input_layer, x)
        if verbose == 1:
            print(model.summary())


if __name__ == "__main__":
    obj = TransferLearnModel()
    obj.get_model()