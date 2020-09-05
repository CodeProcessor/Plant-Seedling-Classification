'''
Created on 9/1/20

@author: dulanj
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from mymodel import MyModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.common import *
from src.preprocessing import Preprocess
from src.transfer_learn_model import TransferLearnModel

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])


class Main():
    def __init__(self):
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(list_of_plants)

        self.train_path = '/home/dulanj/Projects/Kaggle/Plant-Seed/plant-seedlings-classification/train'
        self.test_path = '/home/dulanj/Projects/Kaggle/Plant-Seed/plant-seedlings-classification/test'
        self.pre_pro = Preprocess(self.train_path, self.test_path)

    def get_model(self):
        # return TransferLearnModel.get_model(verbose=1)
        return MyModel.get_model(verbose=1)

    def create_dir_if_not(self, dir_name):
        import os
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def main(self):
        X, y = self.pre_pro.load_data()

        # Encode labels and create classes
        print("Classes: " + str(self.label_encoder.classes_))
        encodeTrainLabels = self.label_encoder.transform(y)

        # Make labels categorical
        TrainLabel = np_utils.to_categorical(encodeTrainLabels)
        num_clases = TrainLabel.shape[1]
        print("Number of classes: " + str(num_clases))
        print(TrainLabel)

        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, random_state=42, shuffle=False)
        k_fold_count = 0
        for train_index, test_index in cv.split(X):
            k_fold_count += 1
            print("K FOLD : {}".format(k_fold_count))
            trainX, testX, trainY, testY = X[train_index], X[test_index], TrainLabel[train_index], TrainLabel[
                test_index]


        # trainX, testX, trainY, testY = train_test_split(X, TrainLabel,
        #                                                 test_size=0.2, random_state=1,
        #                                                 stratify=TrainLabel)

            datagen = ImageDataGenerator(
                zoom_range=0.2,                 # This will randomly zoom the image
                brightness_range=[0.7, 1.3],    # Randomly change the brightness
                horizontal_flip=True,           # This will randomly flip - horizontally
                vertical_flip=True,             # This will randomly flip - vertically
                rotation_range=180,             # Randomly rotate images 0 to 180
                width_shift_range=0.15,          # Randomly shift the images - horizontally
                height_shift_range=0.15          # Randomly shift the images - vertically
            )
            datagen.fit(trainX)

            model = self.get_model()

            epochs = 50
            print("Train shape {} Test shape {}".format(trainY.shape, trainX.shape))
            batch_size = 4
            steps_per_epo = len(trainX) / batch_size
            print("Epoch {} Batch size {} Steps per epoch {}".format(epochs, batch_size, steps_per_epo))

            # from tf.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
            from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

            # learning rate reduction parameters
            learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                        patience=4, verbose=1,
                                                        factor=0.5, min_lr=0.00001)

            # checkpoints
            self.create_dir_if_not('output')
            filepath = "output/weights_best_fold-" + str(k_fold_count) + "_{epoch:02d}-acc{accuracy:.2f}.hdf5"


            checkpoint = ModelCheckpoint(filepath, monitor='accuracy',
                                         verbose=1, save_best_only=True, mode='max')

            filepath = "output/weights.last_auto4.hdf5"


            filepath2 = "output/weights_best_fold-" + str(k_fold_count) + "_{epoch:02d}-val{val_accuracy:.2f}.hdf5"
            checkpoint_all = ModelCheckpoint(filepath2, monitor='val_accuracy',
                                             verbose=1, save_best_only=True, mode='max')

            # all callbacks
            callbacks_list = [learning_rate_reduction, checkpoint_all]

            model.fit(datagen.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=steps_per_epo,
                      epochs=epochs, validation_data=(testX, testY), callbacks=callbacks_list)
            tf.keras.backend.clear_session()

    def get_confusion_matrix(self, best_model_path):
        # Load data
        X, y = self.pre_pro.load_data()

        # Encode labels
        encode_labels = self.label_encoder.transform(y)

        # Make labels categorical
        categorical_labels = np_utils.to_categorical(encode_labels)

        # Load model
        model = self.get_model()
        model.load_weights(best_model_path)

        # PREDICTIONS
        y_predictions = model.predict(X)
        y_class = np.argmax(y_predictions, axis=1)
        y_check = np.argmax(categorical_labels, axis=1)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_check, y_class)
        print(conf_matrix)

    def test(self, best_model_path):
        X, ids = self.pre_pro.load_test_data()
        model = self.get_model()
        model.load_weights(best_model_path)

        pred = model.predict(X)

        # Write result to file
        predNum = np.argmax(pred, axis=1)
        predStr = self.label_encoder.classes_[predNum]

        res = {'file': ids, 'species': predStr}
        df = pd.DataFrame(res)

        from datetime import datetime
        now = datetime.now()
        self.create_dir_if_not('result')
        output_filename = "result/result_{}.csv".format(now.strftime("%Y-%m-%dT%H:%M"))
        df.to_csv(output_filename, index=False)
        print("Results saved to {}".format(output_filename))


if __name__ == "__main__":
    TEST = False
    obj = Main()
    best_model_path = "/home/dulanj/Projects/Kaggle/Plant-Seed/Plant-Seeding-Classification/src/output-mymodel/weights_best_fold-0_45-val0.75.hdf5"
    obj.get_confusion_matrix(best_model_path)

    # if not TEST:
    #     obj.main()
    # else:
    #     best_model_path = "/home/dulanj/Projects/Kaggle/Plant-Seed/Plant-Seeding-Classification/src/output/weights_best_fold-0_45-val0.75.hdf5"
    #     obj.test(best_model_path)
