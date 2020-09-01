'''
Created on 9/1/20

@author: dulanj
'''
import cv2 as cv
import numpy as np


class preprocessing():
    def __init__(self):
        pass

    def main(self, image):
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


if __name__ == "__main__":
    obj = preprocessing()
    obj.main()