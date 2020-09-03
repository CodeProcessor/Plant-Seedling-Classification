'''
Created on 9/1/20

@author: dulanj
'''
from keras.utils import np_utils

from src.mymodel import MyModel
import tensorflow as tf
from sklearn import preprocessing
from src.preprocessing import Preprocess

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
        train_path = '/home/dulanj/Projects/Kaggle/Plant-Seed/plant-seedlings-classification/train'
        test_path = '/home/dulanj/Projects/Kaggle/Plant-Seed/plant-seedlings-classification/test'
        pre_pro = Preprocess(train_path, test_path)
        X, y = pre_pro.load_data()

        # Encode labels and create classes
        print(y)
        le = preprocessing.LabelEncoder()
        le.fit(y)
        print("Classes: " + str(le.classes_))
        encodeTrainLabels = le.transform(y)

        # Make labels categorical
        clearTrainLabel = np_utils.to_categorical(encodeTrainLabels)
        num_clases = clearTrainLabel.shape[1]
        print("Number of classes: " + str(num_clases))
        print(clearTrainLabel)



if __name__ == "__main__":
    obj = Main()
    obj.main()