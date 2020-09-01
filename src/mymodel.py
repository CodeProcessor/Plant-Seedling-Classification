'''
Created on 9/1/20

@author: dulanj
'''
import tensorflow as tf

from src.model import Model


class MyModel(Model):
    def __init__(self):
        pass

    @staticmethod
    def get_model(verbose=1):
        inputs = tf.keras.Input(shape=(120, 120, 1))
        x = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu)(inputs)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu)(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(160, (3, 3), activation=tf.nn.relu)(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu)(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)
        outputs = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        # compile model
        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        if verbose == 1:
            print(model.summary())

        return model


if __name__ == "__main__":
    obj = MyModel()
    obj.model()