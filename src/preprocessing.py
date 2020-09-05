'''
Created on 9/1/20

@author: dulanj
'''
import glob
import os

import cv2 as cv
import numpy as np
from src.common import input_shape


class Preprocess():
    input_shape = (input_shape[0], input_shape[1])

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
                if i > 100:
                    break

        X = np.asarray(X)
        y = np.array(y)

        return X, y

    def load_test_data(self):
        X = []
        ids = []
        for i, file_path in enumerate(glob.glob(self.test_path + '/*')):
            print(file_path)
            img = cv.imread(file_path)
            X.append(self.preprocess_image(img))
            ids.append(os.path.basename(file_path))
        X = np.asarray(X)

        return X, ids

    def preprocess_image(self, image):
        """
        This method will do the pre-processing for a given image

        """
        # Resize the image
        image = cv.resize(image, Preprocess.input_shape)

        # Use Gaussian blur
        blurred_image = cv.GaussianBlur(image, (5, 5), 0)

        # Convert to HSV image
        HSV_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2HSV)

        # Create the mask based on the green values
        lower_green = (25, 40, 50)
        upper_green = (75, 255, 255)
        mask = cv.inRange(HSV_image, lower_green, upper_green)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # Create bool mask
        binary_mask = mask > 0

        # Apply the mask
        clear = np.zeros_like(image, np.uint8)  # Create empty image
        clear[binary_mask] = image[binary_mask]  # Apply boolean mask to the origin image

        # Normalize pixel values
        normalized_image = clear / 255

        return normalized_image

#
# if __name__ == "__main__":
#     obj = preprocessing()
#     obj.main()
