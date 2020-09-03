'''
Created on 9/1/20

@author: dulanj
'''
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from src.mymodel import MyModel
import tensorflow as tf
from sklearn import preprocessing
from src.preprocessing import Preprocess
from src.transfer_learn_model import TransferLearnModel

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_per_process_memory_fraction(0.75)
# tf.config.set_per_process_memory_growth(True)
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])
# config = tf.config.experimental.g


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

        trainX, testX, trainY, testY = train_test_split(X, clearTrainLabel,
                                                        test_size=0.3, random_state=1,
                                                        stratify=clearTrainLabel)


        datagen = ImageDataGenerator(
            rotation_range=180,  # randomly rotate images in the range
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically
            horizontal_flip=True,  # randomly flip images horizontally
            vertical_flip=True  # randomly flip images vertically
        )
        datagen.fit(trainX)

        model = TransferLearnModel.get_model(verbose=0)

        epochs = 5
        print(trainY.shape)
        print(trainX.shape)
        batch_size = 5
        model.fit(datagen.flow(trainX, trainY, batch_size=batch_size),
                  steps_per_epoch=len(trainX) / batch_size, epochs=epochs)



if __name__ == "__main__":
    obj = Main()
    obj.main()