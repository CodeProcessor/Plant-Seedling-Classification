'''
Created on 9/1/20

@author: dulanj
'''
import os

import cv2 as cv
import numpy as np
import glob


class Preprocess():
    input_shape = (70, 70)

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        X = []
        y = []
        for dir_path in glob.glob(self.train_path + '/*'):
            plant_name = os.path.basename(dir_path)

            print(plant_name)
            for i, file_path in enumerate(glob.glob(dir_path + '/*')):
                print(file_path)
                img = cv.imread(file_path)
                X.append(self.preprocess_image(img))
                y.append(plant_name)
                if i > 10:
                    break

        X = np.asarray(X)
        y = np.array(y)

        return X, y

    # def generate_data(directory, batch_size):
    #     """Replaces Keras' native ImageDataGenerator."""
    #     i = 0
    #     file_list = os.listdir(directory)
    #     while True:
    #         image_batch = []
    #         for b in range(batch_size):
    #             if i == len(file_list):
    #                 i = 0
    #                 random.shuffle(file_list)
    #             sample = file_list[i]
    #             i += 1
    #             image = cv2.resize(cv2.imread(sample[0]), INPUT_SHAPE)
    #             image_batch.append((image.astype(float) - 128) / 128)
    #
    #         yield np.array(image_batch)

    def preprocess_image(self, image):
        image = cv.resize(image, Preprocess.input_shape)
        # Use gaussian blur
        blurImg = cv.GaussianBlur(image, (5, 5), 0)

        # Convert to HSV image
        hsvImg = cv.cvtColor(blurImg, cv.COLOR_BGR2HSV)

        # Create mask (parameters - green color range)
        lower_green = (25, 40, 50)
        upper_green = (75, 255, 255)
        mask = cv.inRange(hsvImg, lower_green, upper_green)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # Create bool mask
        bMask = mask > 0

        # Apply the mask
        clear = np.zeros_like(image, np.uint8)  # Create empty image
        clear[bMask] = image[bMask]  # Apply boolean mask to the origin image
        return clear

#
# if __name__ == "__main__":
#     obj = preprocessing()
#     obj.main()