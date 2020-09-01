'''
Created on 9/1/20

@author: dulanj
'''
from src.mymodel import MyModel
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Main():
    def __init__(self):
        pass

    def resnet50(self):
        model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
        print(model.summary())

    def main(self):
        model = MyModel().model()


if __name__ == "__main__":
    obj = Main()
    obj.resnet50()