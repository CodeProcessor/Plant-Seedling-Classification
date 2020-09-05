'''
Created on 9/1/20

@author: dulanj
'''
import tensorflow as tf

from src.common import input_shape
from src.model import Model


class TransferLearnModel(Model):
    def __init__(self):
        pass


def get_model(verbose, type='resnet'):
    # Load base model
    if not type == 'resnet':
        model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(input_shape[0],
                                                                            input_shape[1], input_shape[2]))
    else:
        model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(input_shape[0],
                                                                            input_shape[1], input_shape[2]))
    input_layer = model.inputs
    x = model.layers[-1].output
    # Add Top model
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(12, activation='softmax')(x)
    model = tf.keras.Model(input_layer, x)
    if verbose == 1:
        print(model.summary())

    # Set Optimizer
    opt = tf.keras.optimizers.Adam(lr=0.001, momentum=0.9)
    # Model compile
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


if __name__ == "__main__":
    obj = TransferLearnModel()
    get_model()
