"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

import cv2
import cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def myID() -> np.int:
    
    return 332307073


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    #assign path with the string given in the function
    path = r'{}'.format(filename)
    #read image as gray scale to img
    img = cv2.imread(path,0)
    #if grayscale assign img_final with img
    if representation == 1:
        #return grayscale
        img_final = img
    #else turn grayscale to RGB
    else:
        #make RGB
        img_temp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_final = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
        #return image array normalized
    return img_final.divide(other = 255)

        


def imDisplay(filename: str, representation: int):
    #use function to load image
    img = imReadAndConvert(filename , representation)
    #plot image
    plt.imshow(img)
    plt.title('image displayed:')
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    
    # Store RGB values of all pixels in lists r, g and b
    for row in imgRGB:
        for temp_r, temp_g, temp_b in row:
            #split in to r,g and b lists
            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)
            # makes y, i and q lists 
            y.append(0.30*temp_r + 0.59*temp_g + 0.11*temp_b)
            i.append(0.74*(temp_r-temp_y) - 0.27*(temp_b-temp_y))
            q.append(0.48*(temp_r-temp_y) + 0.41*(temp_b-temp_y))
            #combines y, i and q lists to image
    yiq_img = np.dstack((y,i,q))  
    return yiq_img





def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    
    
    for row in imgYIQ:
        for temp_y, temp_i, temp_q in row:
            #split in to y,i and q lists
            y.append(temp_y)
            i.append(temp_i)
            q.append(temp_q)
            # makes r, g and b lists 
            r.append(temp_y + 0.9468822170900693*temp_i + 0.6235565819861433*temp_q)
            g.append(temp_y - 0.27478764629897834*temp_i - 0.6356910791873801*temp_q)
            b.append(temp_y - 1.1085450346420322*temp_i + 1.7090069284064666*temp_q)
            #combines r, g and b lists to image
    rgb_img = np.dstack((r,g,b))  
    return rgb_img


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    #make histogram and bins to show the amount of pixels with there represpective pixel values
    hist,bins = np.histogram(imgOrig.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    # change pixels to make a more smooth norm
    equ = cv.equalizeHist(img)
    cv.imshow('equ.png',equ)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #show the diffrance
    hist,bins = np.histogram(equ.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(equ.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    
    # not done
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
